"""Regression smoke tests for the benchmark entry points.

These exercise the data loaders and a tiny number of training/projection steps
on each benchmark so that future renames (cf. PR #96) cannot silently break
the scripts. Tests gracefully skip when the in-repo dataset for a benchmark is
not present, so the suite runs in environments without the heavy artifacts.

Heavier end-to-end benchmark sweeps are marked ``@pytest.mark.benchmark`` and
opted out of by default via ``pytest.ini`` (``-m "not benchmark"``). Run them
explicitly with ``uv run pytest -m benchmark``.
"""

from __future__ import annotations

import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from flax.training import train_state

from benchmarks.model import setup_model as setup_qp_model
from benchmarks.QP.load_qp import (
    dc3_dataset_setup,
    non_dc3_dataset_setup,
)
from benchmarks.QP.load_qp import (
    load_data as load_qp_data,
)
from benchmarks.toy_MPC.load_toy_mpc import load_data as load_toy_mpc_data
from benchmarks.toy_MPC.model import setup_model as setup_toy_mpc_model
from pinet import ProjectionInstance, SocConstraint, SocConstraintSpecification

jax.config.update("jax_enable_x64", True)

_QP_DATASETS = pathlib.Path(__file__).parent.parent / "benchmarks" / "QP" / "datasets"
_TOY_MPC_DATASETS = (
    pathlib.Path(__file__).parent.parent / "benchmarks" / "toy_MPC" / "datasets"
)


_SMOKE_HYPERPARAMS: dict[str, Any] = {
    "n_epochs": 1,
    "n_iter_train": 20,
    "n_iter_test": 20,
    "learning_rate": 1.0e-2,
    "sigma": 0.07,
    "omega": 1.7,
    "unroll": False,
    "fpi": False,
    "n_iter_bwd": 10,
    "equilibrate": {
        "max_iter": 0,
        "tol": 1.0e-3,
        "ord": 2.0,
        "col_scaling": False,
        "update_mode": "Gauss",
        "safeguard": False,
    },
    "features_list": [32, 32],
    "activation": "relu",
    "batch_size": 64,
}

_LR = float(_SMOKE_HYPERPARAMS["learning_rate"])  # narrowed for optax.adam


def _require(path: pathlib.Path) -> None:
    if not path.exists():
        pytest.skip(f"required dataset missing: {path}")


def test_simple_qp_dataset_loads():
    """The in-repo SimpleQP small dataset is parseable by the loader."""
    dataset_path = _QP_DATASETS / "SimpleQP_seed42_var100_ineq50_eq50_examples200.npz"
    _require(dataset_path)

    q_mat, _p, a, g_mat, h, x_data, train_loader, _val_loader, _test_loader = (
        non_dc3_dataset_setup(
            use_convex=True,
            problem_seed=42,
            problem_var=100,
            problem_nineq=50,
            problem_neq=50,
            problem_examples=200,
            rng_key=jax.random.PRNGKey(0),
            batch_size=32,
            use_jax_loader=False,
        )
    )

    # Shape contract for downstream consumers (model.py).
    assert q_mat.shape == (1, 100, 100)
    assert a.shape == (1, 50, 100)
    assert g_mat.shape == (1, 50, 100)
    assert h.shape == (1, 50, 1)
    assert x_data.shape == (200, 50, 1)
    assert all(np.isfinite(np.asarray(arr)).all() for arr in (q_mat, a, g_mat, h))

    # The loaders must yield at least one batch with the expected leading dim.
    x_batch, obj_batch = next(iter(train_loader))
    assert x_batch.shape[1:] == (50, 1)
    assert obj_batch.ndim == 1


def test_dc3_simple_dataset_loads():
    """The DC3 simple convex dataset is parseable by the loader."""
    base = _QP_DATASETS / "dc3_random_simple_dataset_var100_ineq50_eq50_ex10000"
    _require(base.with_name(base.name + "train.npz"))

    q_mat, _p, a, g_mat, _h, _x_data, train_loader, _val_loader, _test_loader = (
        dc3_dataset_setup(
            use_convex=True,
            problem_seed=42,
            problem_var=100,
            problem_nineq=50,
            problem_neq=50,
            problem_examples=10000,
            rng_key=jax.random.PRNGKey(0),
            batch_size=128,
            use_jax_loader=True,
        )
    )

    assert q_mat.shape[1] == q_mat.shape[2]
    assert a.shape[2] == q_mat.shape[1]
    assert g_mat.shape[2] == q_mat.shape[1]

    x_batch, obj_batch = next(iter(train_loader))
    assert x_batch.shape[1:] == (a.shape[1], 1)
    # DC3 objectives are (B, 1, 1) after a quadratic-form vmap; just check finite.
    assert np.isfinite(np.asarray(obj_batch)).all()


def test_toy_mpc_dataset_loads():
    """The toy_MPC dataset is parseable by both loader paths."""
    dataset_path = _TOY_MPC_DATASETS / "toy_MPC_seed42_examples10000.npz"
    _require(dataset_path)

    for use_jax_loader in (True, False):
        (
            a,
            _lbxs,
            _ubxs,
            _lbus,
            _ubus,
            _xhat,
            alpha,
            horizon,
            base_dim,
            x_data,
            train_loader,
            _val_loader,
            _test_loader,
            _batched_objective,
        ) = load_toy_mpc_data(
            filepath=str(dataset_path),
            rng_key=jax.random.PRNGKey(0),
            val_split=0.1,
            test_split=0.1,
            batch_size=128,
            use_jax_loader=use_jax_loader,
        )

        assert a.ndim == 3
        assert isinstance(horizon, int) and horizon > 0
        assert isinstance(base_dim, int) and base_dim > 0
        assert alpha > 0
        assert x_data.ndim == 3

        x_batch, obj_batch = next(iter(train_loader))
        assert x_batch.shape[1:] == x_data.shape[1:]
        assert obj_batch.ndim == 1


def test_simple_qp_one_step_finite():
    """One training step on SimpleQP returns finite loss + non-negative CV."""
    dataset_path = _QP_DATASETS / "SimpleQP_seed42_var100_ineq50_eq50_examples200.npz"
    _require(dataset_path)

    torch.manual_seed(0)
    rng = jax.random.PRNGKey(0)
    loader_rng, model_rng = jax.random.split(rng)

    (
        a,
        g_mat,
        h,
        x_data,
        _batched_objective,
        train_loader,
        _val_loader,
        _test_loader,
        batched_loss,
    ) = load_qp_data(
        use_dc3_dataset=False,
        use_convex=True,
        problem_seed=42,
        problem_var=100,
        problem_nineq=50,
        problem_neq=50,
        problem_examples=200,
        rng_key=loader_rng,
        batch_size=64,
        use_jax_loader=False,
        penalty=0.0,
    )

    model, params, _setup_time, train_step = setup_qp_model(
        rng_key=model_rng,
        hyperparameters=_SMOKE_HYPERPARAMS,
        proj_method="pinet",
        a_dyn=a,
        x_data=x_data,
        g_mat=g_mat,
        h=h,
        batched_loss=batched_loss,
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optax.adam(_LR),
    )

    x_batch, _ = next(iter(train_loader))
    loss, state = train_step(state, x_batch[:, :, 0], x_batch)
    assert jnp.isfinite(loss), f"loss must be finite, got {loss}"

    predictions = state.apply_fn(
        {"params": state.params}, x=x_batch[:, :, 0], b=x_batch, test=True
    )
    assert predictions.shape == (x_batch.shape[0], a.shape[2])

    # Constraints should be (very nearly) satisfied after projection.
    eq_cv = jnp.max(jnp.abs(a[0] @ predictions.reshape(-1, a.shape[2], 1) - x_batch))
    assert eq_cv < 1e-4, f"equality CV unexpectedly high: {eq_cv}"


def test_toy_mpc_one_step_finite():
    """One training step on toy_MPC returns a finite loss and respects shapes."""
    dataset_path = _TOY_MPC_DATASETS / "toy_MPC_seed42_examples10000.npz"
    _require(dataset_path)

    torch.manual_seed(0)
    rng = jax.random.PRNGKey(0)
    loader_rng, model_rng = jax.random.split(rng)

    (
        a,
        lbxs,
        ubxs,
        lbus,
        ubus,
        _xhat,
        _alpha,
        _horizon,
        _base_dim,
        x_data,
        train_loader,
        _val_loader,
        _test_loader,
        batched_objective,
    ) = load_toy_mpc_data(
        filepath=str(dataset_path),
        rng_key=loader_rng,
        val_split=0.1,
        test_split=0.1,
        batch_size=64,
        use_jax_loader=True,
    )

    x_full = jnp.concatenate(
        (x_data, jnp.zeros((x_data.shape[0], a.shape[1] - x_data.shape[1], 1))), axis=1
    )
    lb = jnp.concatenate((lbxs, lbus), axis=1)
    ub = jnp.concatenate((ubxs, ubus), axis=1)

    model, params, train_step = setup_toy_mpc_model(
        rng_key=model_rng,
        hyperparameters=_SMOKE_HYPERPARAMS,
        a_dyn=a,
        x_data=x_data,
        b=x_full,
        lb=lb,
        ub=ub,
        batched_objective=batched_objective,
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optax.adam(_LR),
    )

    x_batch, _ = next(iter(train_loader))
    x_batch_full = jnp.concatenate(
        (x_batch, jnp.zeros((x_batch.shape[0], a.shape[1] - x_batch.shape[1], 1))),
        axis=1,
    )
    loss, state = train_step(state, x_batch[:, :, 0], x_batch_full)
    assert jnp.isfinite(loss), f"loss must be finite, got {loss}"


def test_pinet_soc_projection_feasible():
    """SocConstraint produces SOC-feasible points (matches toy_SOC's helper)."""
    m, b_size = 8, 4
    dim = m + 1
    mask_u = jnp.zeros((dim,), dtype=jnp.bool_).at[:m].set(True)
    mask_t = jnp.zeros((dim,), dtype=jnp.bool_).at[m].set(True)
    soc = SocConstraint(socspec=SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t))

    key = jax.random.PRNGKey(0)
    x = jax.random.uniform(key, (b_size, dim, 1), minval=-1.0, maxval=1.0)
    projected = soc.project(ProjectionInstance(x=x)).x
    u_proj = projected[:, :m]
    t_proj = projected[:, m:]
    cv = jnp.maximum(jnp.linalg.norm(u_proj, axis=1, keepdims=True) - t_proj, 0.0)
    assert float(jnp.max(cv)) < 1e-10, "SocConstraint.project must satisfy ||u|| ≤ t"


@pytest.mark.benchmark
def test_dc3_simple_qp_benchmark_progresses():
    """Run a handful of DC3 small QP training steps and check loss decreases."""
    base = _QP_DATASETS / "dc3_random_simple_dataset_var100_ineq50_eq50_ex10000"
    _require(base.with_name(base.name + "train.npz"))

    torch.manual_seed(0)
    rng = jax.random.PRNGKey(0)
    loader_rng, model_rng = jax.random.split(rng)

    hyperparameters = dict(_SMOKE_HYPERPARAMS)
    hyperparameters["n_iter_train"] = 50
    hyperparameters["batch_size"] = 256

    (
        a,
        g_mat,
        h,
        x_data,
        _batched_objective,
        train_loader,
        _val_loader,
        _test_loader,
        batched_loss,
    ) = load_qp_data(
        use_dc3_dataset=True,
        use_convex=True,
        problem_seed=42,
        problem_var=100,
        problem_nineq=50,
        problem_neq=50,
        problem_examples=10000,
        rng_key=loader_rng,
        batch_size=hyperparameters["batch_size"],
        use_jax_loader=True,
        penalty=0.0,
    )

    model, params, _setup_time, train_step = setup_qp_model(
        rng_key=model_rng,
        hyperparameters=hyperparameters,
        proj_method="pinet",
        a_dyn=a,
        x_data=x_data,
        g_mat=g_mat,
        h=h,
        batched_loss=batched_loss,
    )

    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optax.adam(float(hyperparameters["learning_rate"])),
    )

    losses: list[float] = []
    for _ in range(5):
        x_batch, _ = next(iter(train_loader))
        loss, state = train_step(state, x_batch[:, :, 0], x_batch)
        losses.append(float(loss))

    assert all(np.isfinite(losses)), losses
    # Final loss should be no worse than the first by more than a small margin.
    assert losses[-1] <= losses[0] + 1e-3


# ----------------------------------------------------------------------------
# Golden-value regression tests
#
# These pin the test-set metrics produced by a deterministic 3-epoch run with
# production-like hyperparameters (mirror of ``benchmark_small_autotune.yaml``
# and ``toy_MPC.yaml`` with ``n_epochs`` cut to 3 for speed). Tolerances are
# loose enough to absorb floating-point reordering across CPU vendors, but
# tight enough that a real regression (e.g. a broken projection step or a
# silently-skipped equilibration pass) shows up. Numbers were captured from
# the ``main`` branch and confirmed bit-identical-up-to-shuffle on PR #96
# with seed 0.
# ----------------------------------------------------------------------------


_QP_PROD_HYPERPARAMS: dict[str, Any] = {
    "n_epochs": 3,
    "n_iter_train": 50,
    "n_iter_test": 50,
    "learning_rate": 1.0e-2,
    "sigma": 0.06806787572155291,
    "omega": 1.7,
    "unroll": False,
    "fpi": False,
    "n_iter_bwd": 25,
    "equilibrate": {
        "max_iter": 25,
        "tol": 1.0e-3,
        "ord": 2.0,
        "col_scaling": False,
        "update_mode": "Gauss",
        "safeguard": False,
    },
    "features_list": [200, 200],
    "activation": "relu",
    "batch_size": 2048,
}


_TOY_MPC_PROD_HYPERPARAMS: dict[str, Any] = {
    "n_epochs": 3,
    "n_iter_train": 200,
    "n_iter_test": 1000,
    "learning_rate": 3.0e-4,
    "sigma": 0.01,
    "omega": 1.7,
    "unroll": False,
    "fpi": False,
    "n_iter_bwd": 200,
    "raw": False,
    "activation": "relu",
    "equilibrate": {
        "max_iter": 0,
        "tol": 1.0e-3,
        "ord": 2.0,
        "col_scaling": False,
        "update_mode": "Gauss",
        "safeguard": False,
    },
    "features_list": [200, 200],
    "val_split": 0.1,
    "test_split": 0.1,
    "batch_size": 512,
}


def _train_qp(
    state: train_state.TrainState,
    train_step: Any,
    train_loader: Any,
    n_epochs: int,
) -> train_state.TrainState:
    """Run ``n_epochs`` of QP training; return the updated TrainState."""
    for _ in range(n_epochs):
        for x_batch, _ in train_loader:
            _loss, state = train_step(state, x_batch[:, :, 0], x_batch)
    return state


def _eval_qp(
    state: train_state.TrainState,
    loader: Any,
    a: jnp.ndarray,
    g_mat: jnp.ndarray,
    h: jnp.ndarray,
    batched_objective: Any,
) -> dict[str, float]:
    """Evaluate a QP model on one batch and return summary metrics."""
    x_data, obj_ref = next(iter(loader))
    preds = state.apply_fn(
        {"params": state.params}, x=x_data[:, :, 0], b=x_data, test=True
    )
    hcnn_obj = batched_objective(preds)
    rs = jnp.mean((hcnn_obj - obj_ref) / jnp.abs(obj_ref))
    eq_cv_max = jnp.max(
        jnp.abs(a[0] @ preds.reshape(x_data.shape[0], a.shape[2], 1) - x_data)
    )
    ineq_cv_max = jnp.max(
        jnp.maximum(g_mat[0] @ preds.reshape(x_data.shape[0], g_mat.shape[2], 1) - h, 0.0)
    )
    return {
        "rs": float(rs),
        "obj": float(hcnn_obj.mean()),
        "opt_obj": float(obj_ref.mean()),
        "eq_cv_max": float(eq_cv_max),
        "ineq_cv_max": float(ineq_cv_max),
    }


@pytest.mark.benchmark
def test_simple_qp_3epoch_golden_metrics():
    """SimpleQP 3-epoch run reproduces main's test metrics (seed 0)."""
    dataset_path = _QP_DATASETS / "SimpleQP_seed42_var100_ineq50_eq50_examples200.npz"
    _require(dataset_path)

    torch.manual_seed(0)
    rng = jax.random.PRNGKey(0)
    loader_rng, model_rng = jax.random.split(rng)

    (
        a,
        g_mat,
        h,
        x_data,
        batched_objective,
        train_loader,
        _val_loader,
        test_loader,
        batched_loss,
    ) = load_qp_data(
        use_dc3_dataset=False,
        use_convex=True,
        problem_seed=42,
        problem_var=100,
        problem_nineq=50,
        problem_neq=50,
        problem_examples=200,
        rng_key=loader_rng,
        batch_size=_QP_PROD_HYPERPARAMS["batch_size"],
        use_jax_loader=False,
        penalty=0.0,
    )

    model, params, _setup_time, train_step = setup_qp_model(
        rng_key=model_rng,
        hyperparameters=_QP_PROD_HYPERPARAMS,
        proj_method="pinet",
        a_dyn=a,
        x_data=x_data,
        g_mat=g_mat,
        h=h,
        batched_loss=batched_loss,
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optax.adam(float(_QP_PROD_HYPERPARAMS["learning_rate"])),
    )

    state = _train_qp(state, train_step, train_loader, n_epochs=3)
    metrics = _eval_qp(state, test_loader, a, g_mat, h, batched_objective)

    # Golden values captured from main (seed=0, 3 epochs, benchmark_small_autotune).
    assert metrics["rs"] == pytest.approx(0.08965, abs=5e-3), metrics
    assert metrics["obj"] == pytest.approx(-15.23, abs=0.1), metrics
    assert metrics["opt_obj"] == pytest.approx(-16.73449, abs=1e-3), metrics
    assert metrics["eq_cv_max"] < 1e-6, metrics
    assert metrics["ineq_cv_max"] < 1e-3, metrics


@pytest.mark.benchmark
def test_dc3_simple_qp_3epoch_golden_metrics():
    """DC3 small QP 3-epoch run reproduces main's test metrics (seed 0)."""
    base = _QP_DATASETS / "dc3_random_simple_dataset_var100_ineq50_eq50_ex10000"
    _require(base.with_name(base.name + "train.npz"))

    torch.manual_seed(0)
    rng = jax.random.PRNGKey(0)
    loader_rng, model_rng = jax.random.split(rng)

    (
        a,
        g_mat,
        h,
        x_data,
        batched_objective,
        train_loader,
        _val_loader,
        test_loader,
        batched_loss,
    ) = load_qp_data(
        use_dc3_dataset=True,
        use_convex=True,
        problem_seed=42,
        problem_var=100,
        problem_nineq=50,
        problem_neq=50,
        problem_examples=10000,
        rng_key=loader_rng,
        batch_size=_QP_PROD_HYPERPARAMS["batch_size"],
        use_jax_loader=True,
        penalty=0.0,
    )

    model, params, _setup_time, train_step = setup_qp_model(
        rng_key=model_rng,
        hyperparameters=_QP_PROD_HYPERPARAMS,
        proj_method="pinet",
        a_dyn=a,
        x_data=x_data,
        g_mat=g_mat,
        h=h,
        batched_loss=batched_loss,
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optax.adam(float(_QP_PROD_HYPERPARAMS["learning_rate"])),
    )

    state = _train_qp(state, train_step, train_loader, n_epochs=3)
    metrics = _eval_qp(state, test_loader, a, g_mat, h, batched_objective)

    # Golden values captured from main (seed=0, 3 epochs, benchmark_small_autotune).
    assert metrics["rs"] == pytest.approx(0.03238, abs=5e-3), metrics
    assert metrics["obj"] == pytest.approx(-14.551, abs=0.1), metrics
    assert metrics["opt_obj"] == pytest.approx(-15.03721, abs=1e-3), metrics
    assert metrics["eq_cv_max"] < 1e-6, metrics
    assert metrics["ineq_cv_max"] < 0.02, metrics


@pytest.mark.benchmark
def test_toy_mpc_3epoch_golden_metrics():
    """toy_MPC 3-epoch run reproduces main's test mean objective (seed 0)."""
    dataset_path = _TOY_MPC_DATASETS / "toy_MPC_seed42_examples10000.npz"
    _require(dataset_path)

    torch.manual_seed(0)
    rng = jax.random.PRNGKey(0)
    loader_rng, model_rng = jax.random.split(rng)

    (
        a,
        lbxs,
        ubxs,
        lbus,
        ubus,
        _xhat,
        _alpha,
        _horizon,
        _base_dim,
        x_data,
        train_loader,
        _val_loader,
        test_loader,
        batched_objective,
    ) = load_toy_mpc_data(
        filepath=str(dataset_path),
        rng_key=loader_rng,
        val_split=_TOY_MPC_PROD_HYPERPARAMS["val_split"],
        test_split=_TOY_MPC_PROD_HYPERPARAMS["test_split"],
        batch_size=_TOY_MPC_PROD_HYPERPARAMS["batch_size"],
        use_jax_loader=True,
    )
    x_full = jnp.concatenate(
        (x_data, jnp.zeros((x_data.shape[0], a.shape[1] - x_data.shape[1], 1))), axis=1
    )
    lb = jnp.concatenate((lbxs, lbus), axis=1)
    ub = jnp.concatenate((ubxs, ubus), axis=1)

    model, params, train_step = setup_toy_mpc_model(
        rng_key=model_rng,
        hyperparameters=_TOY_MPC_PROD_HYPERPARAMS,
        a_dyn=a,
        x_data=x_data,
        b=x_full,
        lb=lb,
        ub=ub,
        batched_objective=batched_objective,
    )
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optax.adam(float(_TOY_MPC_PROD_HYPERPARAMS["learning_rate"])),
    )

    for _ in range(3):
        for x_batch, _ in train_loader:
            x_batch_full = jnp.concatenate(
                (
                    x_batch,
                    jnp.zeros((x_batch.shape[0], a.shape[1] - x_batch.shape[1], 1)),
                ),
                axis=1,
            )
            _loss, state = train_step(state, x_batch[:, :, 0], x_batch_full)

    # Evaluate on the test loader (single batch).
    x_test, opt_obj = next(iter(test_loader))
    x_test_full = jnp.concatenate(
        (x_test, jnp.zeros((x_test.shape[0], a.shape[1] - x_test.shape[1], 1))), axis=1
    )
    preds = state.apply_fn(
        {"params": state.params}, x_test[:, :, 0], x_test_full, test=True
    )
    test_obj = float(batched_objective(preds).mean())
    opt_obj_mean = float(opt_obj.mean())

    # Constraint checks (should be machine-zero after projection).
    eq_cv = float(jnp.max(jnp.abs(a[0] @ preds.reshape(-1, a.shape[2], 1) - x_test_full)))
    ineq_cv = float(
        jnp.maximum(
            jnp.max(preds.reshape(-1, a.shape[2], 1) - ub),
            jnp.max(lb - preds.reshape(-1, a.shape[2], 1)),
        )
    )

    # Golden values captured from main (seed=0, 3 epochs, toy_MPC.yaml).
    # Test obj on main was 1378.53; PR-with-shim hits 1378.39; tolerance ±10
    # absorbs the small shuffle-order difference and minor FP reordering on CI.
    assert test_obj == pytest.approx(1378.5, abs=10.0), (test_obj, opt_obj_mean)
    assert opt_obj_mean == pytest.approx(1210.27, abs=1e-1), (test_obj, opt_obj_mean)
    assert eq_cv < 1e-6, eq_cv
    assert ineq_cv < 1e-6, ineq_cv
