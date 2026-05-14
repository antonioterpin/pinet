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
