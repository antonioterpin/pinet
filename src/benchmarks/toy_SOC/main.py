"""This module implements the solver for random second-order cone parametric problems."""

# %% Imports
import datetime
import pathlib

import cvxpy as cp
import jax
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
from jax import config as jconf
from jax import jit
from jax import numpy as jnp
from jax import random as jrnd
from jax import value_and_grad
from tqdm import tqdm

from benchmarks.toy_SOC.generate_socp import (
    constraint_violation_eq,
    constraint_violation_soc,
    generate_problem,
    objective,
    rand_sparse_mask,
    relative_suboptimality,
)
from benchmarks.toy_SOC.model import HardConstrainedMLP
from benchmarks.toy_SOC.projection_layer import build_projection_layer
from src.tools.utils import GracefulShutdown, Logger, load_configuration

# Use 64 bit precision for numerical stability
jconf.update("jax_enable_x64", True)

# %% Setup
CONFIG = "toy_SOC"
config_path = (
    pathlib.Path(__file__).parent.parent.resolve() / "configs" / (CONFIG + ".yaml")
)
hyperparameters = load_configuration(config_path)
# %%
SEED = 1
# Problem dimensions
n = 250
m = 250
sparsity = 0.01
nowstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"n{n}_m{m}_{nowstamp}"
# Key
key = jrnd.PRNGKey(SEED)
ACTIVATION = getattr(nn, hyperparameters["activation"], None)
if ACTIVATION is None:
    raise ValueError(f"Unknown activation: {hyperparameters['activation']}")
LAYERS = hyperparameters["features_list"]

keyA, key = jrnd.split(key)
A = rand_sparse_mask(keyA, (m, n), sparsity=sparsity)

# %% CV and RS


def print_stats(
    x: jnp.ndarray, s: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, xstar: jnp.ndarray
):
    """Print the statistics of the solution.

    Args:
        x (jnp.ndarray): Primal solution, shape (B, n, 1).
        s (jnp.ndarray): Dual solution, shape (B, m, 1).
        b (jnp.ndarray): Right-hand side of the equality constraints, shape (B, m, 1).
        c (jnp.ndarray): Coefficients for the objective function, shape (B, n, 1).
        xstar (jnp.ndarray): Optimal primal solution, shape (B, n, 1).
    """
    cv_eq = constraint_violation_eq(A, x, s, b)
    cv_soc = constraint_violation_soc(s)
    rs = relative_suboptimality(x, xstar, c)

    print("=========== Solution statistics ===========")
    # mean, std, max, min
    print(
        f"""CV (Ax = b): {jnp.mean(cv_eq):.15f} ± {jnp.std(cv_eq):.15f}
          in [{jnp.min(cv_eq):.15f}, {jnp.max(cv_eq):.15f}]"""
    )
    print(
        f"""CV (SOC): {jnp.mean(cv_soc):.15f} ± {jnp.std(cv_soc):.15f}
          in [{jnp.min(cv_soc):.15f}, {jnp.max(cv_soc):.15f}]"""
    )
    print(
        f"""RS: {jnp.mean(rs):.15f} ± {jnp.std(rs):.15f}
          in [{jnp.min(rs):.15f}, {jnp.max(rs):.15f}]"""
    )


# %% Validate the problem
B = 1024
# Symbolic problem
b, c, xstar, sstar = generate_problem(key, A, B)

# %% CVXPY
A_np = np.asarray(A)
x_var = cp.Variable(n)
s_var = cp.Variable(m)
b_par = cp.Parameter(m)
c_par = cp.Parameter(n)

constraints = [A_np @ x_var + s_var == b_par, cp.SOC(s_var[-1], s_var[:-1])]

problem = cp.Problem(cp.Minimize(c_par @ x_var), constraints)
x_sol = []
s_sol = []
for i in tqdm(range(B)):
    b_par.value = np.asarray(b[i]).ravel()  # shape (m,)
    c_par.value = np.asarray(c[i]).ravel()  # shape (n,)

    problem.solve(solver=cp.SCS, verbose=False, eps_abs=1e-9, eps_rel=1e-9)

    if x_var.value is None:
        raise RuntimeError(f"sample {i}: {problem.status}")
    x_sol.append(x_var.value.reshape(n, 1))
    s_sol.append(s_var.value.reshape(m, 1))

x_cvxpy = jnp.asarray(x_sol).reshape(B, n, 1)
s_cvxpy = jnp.asarray(s_sol).reshape(B, m, 1)

# Print the statistics of the solution
print_stats(x_cvxpy, s_cvxpy, b, c, xstar)

# %% Use our solver
n_iter_forward = hyperparameters["n_iter_train"]
n_iter_backward = hyperparameters["n_iter_bwd"]
sigma = hyperparameters["sigma"]
omega = hyperparameters["omega"]

Aaug = jnp.concatenate((A, jnp.eye(m)), axis=1)
assert Aaug.shape == (
    m,
    m + n,
), f"Augmented matrix A should have shape ({m}, {m + n}), instead: {Aaug.shape}"

project = build_projection_layer(
    Aaug, sigma, omega, n, n_iter_forward, n_iter_backward, use_custom_vjp=True
)

# %% Test the projection
# To test the correctness of the projection, we can sample random points,
# project them, and check if the result has no constraint violation.


def test_projection(
    b: jnp.ndarray, c: jnp.ndarray, xstar: jnp.ndarray, sstar: jnp.ndarray
):
    """Test the projection function on random samples.

    Args:
        b (jnp.ndarray): Right-hand side of the equality constraints, shape (B, m, 1).
        c (jnp.ndarray): Coefficients for the objective function, shape (B, n, 1).
        xstar (jnp.ndarray): Optimal primal solution, shape (B, n, 1).
        sstar (jnp.ndarray): Optimal dual solution, shape (B, m, 1).
    """
    n_samples = b.shape[0]
    yraw = jrnd.uniform(key, (n_samples, n + m, 1))

    x = yraw[:, :n]
    s = yraw[:, n:]
    cv_eq_raw = constraint_violation_eq(A, x, s, b)
    cv_soc_raw = constraint_violation_soc(s)
    if jnp.all(cv_eq_raw < 1e-6) and jnp.all(cv_soc_raw < 1e-6):
        print(f"Sample {i}: No constraint violation in the raw samples.")

    cv_eq_opt = constraint_violation_eq(A, xstar, sstar, b)
    cv_soc_opt = constraint_violation_soc(sstar)
    if jnp.any(cv_eq_opt > 1e-6) or jnp.any(cv_soc_opt > 1e-6):
        print(f"Optimal sample: {cv_eq_opt.max()=}, {cv_soc_opt.max()=}")

    # Project the point
    y, _ = project(jnp.zeros_like(yraw), yraw, b)
    x = y[:, :n]
    s = y[:, n:]

    # Check the constraint violation
    cv_eq = constraint_violation_eq(A, x, s, b)
    cv_eq_soc = constraint_violation_soc(s)
    if (
        jnp.any(cv_eq > 1e-6)
        or jnp.any(cv_eq_soc > 1e-6)
        or jnp.isnan(cv_eq).any()
        or jnp.isnan(cv_eq_soc).any()
    ):
        print(f"Projection failed: {cv_eq.max()=}, {cv_eq_soc.max()=}")
        print_stats(x, s, b, c, xstar.reshape(-1, n, 1))
    print("All projections passed.")


test_projection(b, c, xstar, sstar)


# %% Train the MLP
BATCH_SIZE = hyperparameters["batch_size"]
N_EPOCHS = hyperparameters["n_epochs"]
LEARNING_RATE = hyperparameters["learning_rate"]
key_train, key_init = jrnd.split(key)


# Batcher
def make_batch(key: jax.random.PRNGKey, batch_size: int = BATCH_SIZE):
    """Generate a batch of random problems.

    Args:
        key (jax.random.PRNGKey): Random key for generating the batch.
        batch_size (int): Number of problem instances in the batch.

    Returns:
        tuple:
            - dict: Input data containing "b" and "c".
            - jnp.ndarray: Optimal primal solution, shape (B, n, 1).
            - jnp.ndarray: Optimal dual solution, shape (B, m, 1).
    """
    key_prob, key = jrnd.split(key)
    b, c, xstar, sstar = generate_problem(key_prob, A, batch_size)
    return {
        "input": {"b": b, "c": c},
        "xstar": xstar,
        "sstar": sstar,
    }, key


# %% Initialize the model
model = HardConstrainedMLP(
    activation=ACTIVATION, layers=LAYERS, project=project, m=m, n=n
)

# Sample one batch only to create shapes for initialisation
batch, key = make_batch(key_init, batch_size=1)
key, key_init = jrnd.split(key)
params = model.init(key_init, batch["input"])

tx = optax.adam(LEARNING_RATE)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# %% Training
@jit
def loss_fn(params: dict, input: dict):
    """Compute the loss function and auxiliary values.

    Args:
        params (dict): Model parameters.
        input (dict): Input data containing "b" and "c".

    Returns:
        tuple:
            - loss (jnp.ndarray): Mean objective value.
            - aux (tuple): Auxiliary values containing constraint violations.
    """
    c = input["c"]
    pred = model.apply(params, input)
    x = pred[:, :n]
    s = pred[:, n:]
    objective_value = objective(x, c)
    return jnp.mean(objective_value), (x, s)


@jit
def train_step(state: train_state.TrainState, batch: dict):
    """Perform a single training step.

    Args:
        state (TrainState): Current state of the model.
        batch (dict): Input data containing "b" and "c".

    Returns:
        tuple:
            - state (TrainState): Updated state of the model.
            - loss (jnp.ndarray): Loss value after the step.
            - aux (tuple): Auxiliary values containing constraint violations.
    """
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params, batch["input"])
    state = state.apply_gradients(grads=grads)
    return state, loss, aux


# Training loop
update_every = 20
with (
    Logger(run_name=run_name, project_name="toy_SOC") as data_logger,
    GracefulShutdown("Stop detected, finish epoch...") as g,
):
    data_logger.run.config.update(hyperparameters)
    for epoch in (pbar := tqdm(range(1, N_EPOCHS + 1))):
        epoch_losses = []
        key_train, key = jrnd.split(key_train)
        batch, key_train = make_batch(key)
        state, l, (x, s) = train_step(state, batch)
        cv_eq = constraint_violation_eq(A, x, s, batch["input"]["b"])
        cv_soc = constraint_violation_soc(s)
        rs = relative_suboptimality(x, batch["xstar"], batch["input"]["c"])
        data_logger.log(
            epoch,
            {
                "loss": l,
                "avg_rs": rs.mean(),
                "max_rs": rs.max(),
                "eq_cv": cv_eq.max(),
                "cv_soc": cv_soc.max(),
            },
        )
        if epoch % update_every == 0 or epoch == 1:
            pbar.set_description(f"Train Loss: {l:.5f}")
            pbar.set_postfix(
                {
                    "CV Eq": f"{cv_eq.max():.4e}",
                    "CV SOC": f"{cv_soc.max():.4e}",
                    "RS": f"{rs.mean():.4e}",
                }
            )

# %% Test
batch_size_test = 1024
key_test, key = jrnd.split(key_train)
val_batch, _ = make_batch(key_test, batch_size=batch_size_test)
pred_val = model.apply(state.params, val_batch["input"])
b = val_batch["input"]["b"]

x_pred = pred_val[:, :n]
s_pred = pred_val[:, n:]

print_stats(
    x_pred, s_pred, val_batch["input"]["b"], val_batch["input"]["c"], val_batch["xstar"]
)

# %%
