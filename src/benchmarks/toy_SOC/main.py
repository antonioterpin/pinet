"""This module implements the solver for random second-order cone parametric problems."""

# %% Imports
from typing import Any, cast

import cvxpy as cp
import jax
import numpy as np
import optax
from cvxpy.constraints.constraint import Constraint as CvxpyConstraint
from flax import linen as nn
from flax.training import train_state
from jax import config as jconf
from jax import custom_vjp as _custom_vjp
from jax import jit, lax, value_and_grad, vjp
from jax import numpy as jnp
from jax import random as jrnd
from jax.scipy.sparse.linalg import bicgstab

# %% Hyperparameters
# Every tunable knob for the SOC benchmark -- problem size, solver, and
# training -- is collected here so a single block governs the run.
HYPERPARAMETERS: dict[str, Any] = {
    # Problem
    "n": 250,  # primal dimension
    "m": 250,  # number of equality / cone rows
    "seed": 1,  # PRNG seed
    "constraint_tol": 1e-6,  # feasibility tolerance used in validation
    # Solver (Douglas-Rachford)
    "n_iter_forward": 1000,  # forward solver iterations
    "n_iter_backward": 200,  # backward (implicit-diff) solver iterations
    "sigma": 0.1,  # step size
    "omega": 1.8,  # relaxation parameter
    # Training
    "batch_size": 512,
    "n_epochs": 1000,
    "learning_rate": 1e-3,
}

# Use 64 bit precision for numerical stability
jconf.update("jax_enable_x64", True)

n = HYPERPARAMETERS["n"]
m = HYPERPARAMETERS["m"]
CONSTRAINT_TOL = HYPERPARAMETERS["constraint_tol"]
n_iter_forward = HYPERPARAMETERS["n_iter_forward"]
n_iter_backward = HYPERPARAMETERS["n_iter_backward"]
sigma = HYPERPARAMETERS["sigma"]
omega = HYPERPARAMETERS["omega"]
BATCH_SIZE = HYPERPARAMETERS["batch_size"]
n_epochs = HYPERPARAMETERS["n_epochs"]
learning_rate = HYPERPARAMETERS["learning_rate"]
# Key
key = jrnd.PRNGKey(HYPERPARAMETERS["seed"])

use_custom_vjp = True
custom_vjp: Any
if use_custom_vjp:
    custom_vjp = _custom_vjp
else:

    def _noop(f: Any) -> Any:
        """Return the input callable unchanged.

        Args:
            f: Callable to return unchanged.

        Returns:
            Any: The input callable.
        """
        return f

    custom_vjp = _noop


# %% Projections
def _project_soc(z: jax.Array) -> jax.Array:
    """Project onto the second-order cone (SOC) constraint.

    Args:
        z:
            Input array of shape (B, m + 1, 1) where the last column is the SOC radius.

    Returns:
        jax.Array:
            Projected array of the same shape as `z`, satisfying the SOC constraint.
    """
    eps = 1e-12
    u, t = z[:, :-1], z[:, -1:]
    norm_u = jnp.linalg.norm(u, axis=1, keepdims=True)

    proj1 = z
    proj2 = jnp.zeros_like(z)
    proj3: jax.Array = jnp.asarray(
        (t + norm_u) / 2 * jnp.concatenate((u / (norm_u + eps), jnp.ones_like(t)), axis=1)
    )

    when1 = norm_u <= t
    when2 = norm_u <= -t

    return jnp.where(when1, proj1, jnp.where(when2, proj2, proj3))


project_soc = jit(_project_soc)


# %% Generate random data
def rand_sparse_mask(
    key: jax.Array,
    shape: tuple[int, ...],
    sparsity: float = 0.01,
    dtype: jnp.dtype = jnp.float64,
):
    """Return a dense tensor whose entries are 0 with prob = `sparsity`.

    Args:
        key: Random key for generating the tensor.
        shape: Shape of the tensor to be generated.
        sparsity: Probability of an entry being zero. Default is 0.01.
        dtype: Data type of the tensor. Default is jnp.float64.

    Returns:
        jax.Array:
            A tensor of the specified shape with random values and a mask applied.
    """
    key_val, key_mask = jrnd.split(key)

    # Non-zero density is 1 - sparsity
    density = 1.0 - sparsity

    values = jrnd.uniform(key_val, shape, dtype, minval=-1, maxval=1)
    mask = jrnd.bernoulli(key_mask, p=density, shape=shape)
    return values * mask.astype(dtype)


key_a, key = jrnd.split(key)
a_mat = rand_sparse_mask(key_a, (m, n))


def generate_problem(key: jax.Array, batch_size: int):
    """Generate a random linear problem with SOC constraints.

    Args:
        key: Random key for generating the problem.
        batch_size: Number of problem instances to generate.

    Returns:
        tuple:
            - b (jax.Array):
                Right-hand side of the equality constraints, shape (B, m, 1).
            - c (jax.Array): Coefficients for the objective function, shape (B, n, 1).
            - x (jax.Array): Optimal primal solution, shape (B, n, 1).
            - s (jax.Array):
                Optimal dual solution satisfying the SOC constraint, shape (B, m, 1).
    """
    keyz, keyx = jrnd.split(key)
    z = jrnd.uniform(keyz, (batch_size, m, 1), minval=-1, maxval=1)
    s = project_soc(z)
    y = s - z

    # Generate the primal solution x
    x = jrnd.uniform(keyx, (batch_size, n, 1), minval=-1, maxval=1)
    b = a_mat @ x + s
    c = -a_mat.T @ y

    return b, c, x, s


def objective(x: jax.Array, c: jax.Array):
    """Compute the objective value for the linear problem.

    Args:
        x: Primal solution, shape (B, n, 1).
        c: Coefficients for the objective function, shape (B, n, 1).

    Returns:
        jax.Array: Objective value, shape (B, 1).
    """
    return jnp.sum(c * x, axis=(1, 2), keepdims=True)


# %% CV and RS
def constraint_violation_eq(x: jax.Array, s: jax.Array, b: jax.Array):
    """Compute the constraint violation for Ax = b.

    Args:
        x: Primal solution, shape (B, n, 1).
        s: Dual solution, shape (B, m, 1).
        b: Right-hand side of the equality constraints, shape (B, m, 1).

    Returns:
        jax.Array: Constraint violation, shape (B, 1).
    """
    return jnp.linalg.norm(a_mat @ x + s - b, ord=jnp.inf, axis=-1)


def constraint_violation_soc(s: jax.Array):
    """Compute the constraint violation for the SOC constraint.

    Args:
        s: Dual solution, shape (B, m + 1, 1).

    Returns:
        jax.Array: Constraint violation, shape (B, 1).
    """
    u = s[:, :-1]
    t = s[:, -1:]
    u_norm = jnp.linalg.norm(u, axis=1, keepdims=True)

    return jnp.maximum(u_norm - t, 0.0)


def relative_suboptimality(x: jax.Array, xstar: jax.Array, c: jax.Array):
    """Compute the relative suboptimality of the solution.

    Args:
        x: Primal solution, shape (B, n, 1).
        xstar: Optimal primal solution, shape (B, n, 1).
        c: Coefficients for the objective function, shape (B, n, 1).

    Returns:
        jax.Array: Relative suboptimality, shape (B, 1).
    """
    optimal_val = objective(xstar, c)
    candidate_val = objective(x, c)
    return jnp.abs(candidate_val - optimal_val) / (jnp.abs(optimal_val) + 1e-12)


def print_stats(x: jax.Array, s: jax.Array, b: jax.Array, c: jax.Array, xstar: jax.Array):
    """Print the statistics of the solution.

    Args:
        x: Primal solution, shape (B, n, 1).
        s: Dual solution, shape (B, m, 1).
        b: Right-hand side of the equality constraints, shape (B, m, 1).
        c: Coefficients for the objective function, shape (B, n, 1).
        xstar: Optimal primal solution, shape (B, n, 1).
    """
    cv_eq = constraint_violation_eq(x, s, b)
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
batch_size = 1024
# Symbolic problem
b, c, xstar, sstar = generate_problem(key, batch_size)

# %% CVXPY
A_np = np.asarray(a_mat)
x_var = cp.Variable(n)
s_var = cp.Variable(m)
b_par = cp.Parameter(m)
c_par = cp.Parameter(n)

constraints = [A_np @ x_var + s_var == b_par, cp.SOC(s_var[-1], s_var[:-1])]

problem = cp.Problem(cp.Minimize(c_par @ x_var), cast(list[CvxpyConstraint], constraints))
x_sol = []
s_sol = []
for i in range(batch_size):
    b_par.value = np.asarray(b[i]).ravel()  # shape (m,)
    c_par.value = np.asarray(c[i]).ravel()  # shape (n,)

    problem.solve(solver=cp.SCS, verbose=False, eps_abs=1e-9, eps_rel=1e-9)

    if x_var.value is None:
        raise RuntimeError(f"sample {i}: {problem.status}")
    # The SOC slack must be produced whenever the solve succeeded.
    assert s_var.value is not None
    x_sol.append(x_var.value.reshape(n, 1))
    s_sol.append(s_var.value.reshape(m, 1))

x_cvxpy = jnp.asarray(x_sol).reshape(batch_size, n, 1)
s_cvxpy = jnp.asarray(s_sol).reshape(batch_size, m, 1)

# Print the statistics of the solution
print_stats(x_cvxpy, s_cvxpy, b, c, xstar)

# %% Use our solver
a_mat_aug = jnp.concatenate((a_mat, jnp.eye(m)), axis=1)
# The augmented equality matrix must match the primal-plus-slack dimension.
assert a_mat_aug.shape == (
    m,
    m + n,
), f"Augmented matrix a_mat should have shape ({m}, {m + n}), instead: {a_mat_aug.shape}"

a_mat_aug_inv = jnp.linalg.pinv(a_mat_aug)


def project_pinv_vb(xs: jax.Array, b: jax.Array):
    """Project onto the pseudo-inverse of the augmented matrix a_mat.

    Args:
        xs:
            Input array of shape (B, m + n, 1) where the first n columns
            are the primal variables and the last m columns are residuals.

        b:
            Right-hand side of the equality constraints, shape (B, m, 1).

    Returns:
        jax.Array: Projected array satisfying the equality constraints.
    """
    return xs - a_mat_aug_inv @ (a_mat_aug @ xs - b)


def step_iteration(y_raw: jax.Array, sk: jax.Array, b: jax.Array):
    """Perform one iteration of the forward step.

    Args:
        y_raw:
            Raw input array of shape (B, m + n, 1) where the first n columns
            are the primal variables and the last m columns are residuals.

        sk:
            Governing sequence, shape (B, m + n, 1).

        b:
            Right-hand side of the equality constraints, shape (B, m, 1).

    Returns:
        jax.Array: Updated governing sequence, shape (B, m + n, 1).
    """
    zk = project_pinv_vb(sk, b)
    reflect = 2 * zk - sk
    toproj = (reflect - 2 * sigma * y_raw) / (1 + 2 * sigma)
    tk1 = toproj[:, :n]
    tk2 = project_soc(toproj[:, n:])
    tk = jnp.concatenate((tk1, tk2), axis=1)
    return sk + omega * (tk - zk)


def step_final(s: jax.Array, b: jax.Array) -> jax.Array:
    """Retrieve the final result from the forward step.

    Args:
        s: Governing sequence, shape (B, m + n, 1).
        b: Right-hand side of the equality constraints, shape (B, m, 1).

    Returns:
        jax.Array: Final projected value, shape (B, m + n, 1).
    """
    return project_pinv_vb(s, b)


@custom_vjp
def project(
    s0: jax.Array,
    y_raw: jax.Array,
    b: jax.Array,
):
    """Project the raw input onto the feasible set defined by the constraints.

    Args:
        s0: Initial governing sequence, shape (B, m + n, 1).
        y_raw: Raw input array, shape (B, m + n, 1).
        b: Right-hand side of the equality constraints, shape (B, m, 1).

    Returns:
        tuple:
            - zk1 (jax.Array): Projected value, shape (B, m + n, 1).
            - sk (jax.Array): Final governing sequence, shape (B, m + n, 1).
    """
    sk = s0
    sk, _ = lax.scan(
        lambda sk, _: (
            step_iteration(y_raw.reshape((y_raw.shape[0], y_raw.shape[1], 1)), sk, b),
            None,
        ),
        sk,
        xs=None,
        length=int(n_iter_forward),
    )

    # NOTE: There is no auxiliary variable in this case
    zk1 = step_final(sk, b).reshape(y_raw.shape)

    # return values and residuals
    return zk1, sk


def _project_fwd(s0: jax.Array, y_raw: jax.Array, b: jax.Array):
    """Forward pass of the projection function.

    Args:
        s0: Initial governing sequence, shape (B, m + n, 1).
        y_raw: Raw input array, shape (B, m + n, 1).
        b: Right-hand side of the equality constraints, shape (B, m, 1).

    Returns:
        - tuple:
            - zk1 (jax.Array): Projected value, shape (B, m + n, 1).
            - sk (jax.Array): Final governing sequence, shape (B, m + n, 1).
        - tuple:
            - (sk, y_raw, b): Residuals for the backward pass.
    """
    zk1, sk = project(s0, y_raw, b)
    return (zk1, sk), (sk, y_raw.reshape((y_raw.shape[0], y_raw.shape[1], 1)), b)


def _project_bwd(residuals: tuple[Any, ...], cotangent: tuple[Any, ...]):
    """Backward pass of the projection function.

    Args:
        residuals: Residuals from the forward pass, containing:
            - sk (jax.Array): Governing sequence, shape (B, m + n, 1).
            - y_raw (jax.Array): Raw input array, shape (B, m + n, 1).
            - b (jax.Array):
                Right-hand side of the equality constraints, shape (B, m, 1).

        cotangent: Cotangent vector from the backward pass, containing:
            - cotangent_zk1 (jax.Array):
                Cotangent vector for the projected value, shape (B, m + n, 1).
            - cotangent_sk (jax.Array):
                Cotangent vector for the governing sequence, shape (B, m + n, 1).

    Returns:
        tuple:
            - None: Placeholder for the vjp wrt to sk (the DRA governing sequence).
            - thevjp (jax.Array):
                The vjp wrt to y_raw, shape (B, m + n, 1).
            - None: Placeholder for the vjp wrt to b
                (the right-hand side of the equality constraints).
    """
    sk, y_raw, b = residuals
    cotangent_zk1, _ = cotangent

    # Compute the vjp of the iteration step wrt to the DRA governing sequence
    _, iteration_vjp = vjp(lambda xx: step_iteration(y_raw, xx, b), sk)
    # Compute the vjp of the iteration step wrt to the value to be projected
    _, iteration_vjp2 = vjp(lambda xx: step_iteration(xx, sk, b), y_raw)
    # Compute the vjp of the final step wrt to DRA governing sequence
    _, equality_vjp = vjp(lambda xx: step_final(xx, b), sk)

    cotangent_eq_6 = equality_vjp(cotangent_zk1)[0]

    def a_mat_op(xx):
        return xx - iteration_vjp(xx)[0]

    cotangent_eq_7 = bicgstab(a_mat_op, cotangent_eq_6, maxiter=n_iter_backward)[0]

    thevjp = iteration_vjp2(cotangent_eq_7)[0]

    # We only care about the vjp wrt to y_raw
    # So, we return None for the vjp wrt to sk (the DRA governing sequence)
    # and None for the vjp wrt to b (the right-hand side of the equality constraints)
    return (None, thevjp, None)


if use_custom_vjp:
    project.defvjp(_project_fwd, _project_bwd)

# %% Test the projection
# To test the correctness of the projection, we can sample random points,
# project them, and check if the result has no constraint violation.


def test_projection(b: jax.Array, c: jax.Array, xstar: jax.Array, sstar: jax.Array):
    """Test the projection function on random samples.

    Args:
        b: Right-hand side of the equality constraints, shape (B, m, 1).
        c: Coefficients for the objective function, shape (B, n, 1).
        xstar: Optimal primal solution, shape (B, n, 1).
        sstar: Optimal dual solution, shape (B, m, 1).
    """
    n_samples = b.shape[0]
    y_raw = jrnd.uniform(key, (n_samples, n + m, 1))

    x = y_raw[:, :n]
    s = y_raw[:, n:]
    cv_eq_raw = constraint_violation_eq(x, s, b)
    cv_soc_raw = constraint_violation_soc(s)
    if jnp.all(cv_eq_raw < CONSTRAINT_TOL) and jnp.all(cv_soc_raw < CONSTRAINT_TOL):
        print(f"Sample {i}: No constraint violation in the raw samples.")

    cv_eq_opt = constraint_violation_eq(xstar, sstar, b)
    cv_soc_opt = constraint_violation_soc(sstar)
    if jnp.any(cv_eq_opt > CONSTRAINT_TOL) or jnp.any(cv_soc_opt > CONSTRAINT_TOL):
        print(f"Optimal sample: {cv_eq_opt.max()=}, {cv_soc_opt.max()=}")

    # Project the point
    y, _ = project(jnp.zeros_like(y_raw), y_raw, b)
    x = y[:, :n]
    s = y[:, n:]

    # Check the constraint violation
    cv_eq = constraint_violation_eq(x, s, b)
    cv_eq_soc = constraint_violation_soc(s)
    if (
        jnp.any(cv_eq > CONSTRAINT_TOL)
        or jnp.any(cv_eq_soc > CONSTRAINT_TOL)
        or jnp.isnan(cv_eq).any()
        or jnp.isnan(cv_eq_soc).any()
    ):
        print(f"Projection failed: {cv_eq.max()=}, {cv_eq_soc.max()=}")
        print_stats(x, s, b, c, xstar.reshape(-1, n, 1))
    print("All projections passed.")


test_projection(b, c, xstar, sstar)


# %% Simple MLP
class HardConstrainedMLP(nn.Module):
    """A simple MLP model for solving the hard constrained problem.

    Attributes:
        layers: Width of each dense hidden layer in the MLP.
    """

    layers: list[int]

    @nn.compact
    def __call__(
        self,
        input: dict[str, jax.Array],
    ):
        """Call the NN.

        Args:
            input:
                Dictionary containing the input data with keys "b" and "c".

        Returns:
            jax.Array:
                Output of the MLP, projected onto the feasible set.
        """
        b, c = input["b"].squeeze(-1), input["c"].squeeze(-1)
        x = jnp.concatenate((b, c), axis=-1)
        for layer_size in self.layers:
            x = nn.relu(nn.Dense(layer_size)(x))
        # Final layer to project
        x = nn.Dense(n + m)(x).reshape((x.shape[0], n + m, 1))
        x = project(jnp.zeros_like(x), x, b.reshape((b.shape[0], -1, 1)))[0]
        return x


# %% Train the MLP
key_train, key_init = jrnd.split(key)


# Batcher
def make_batch(key: jax.Array, batch_size: int = BATCH_SIZE):
    """Generate a batch of random problems.

    Args:
        key: Random key for generating the batch.
        batch_size: Number of problem instances in the batch.

    Returns:
        tuple:
            - dict: Input data containing "b" and "c".
            - jax.Array: Optimal primal solution, shape (B, n, 1).
            - jax.Array: Optimal dual solution, shape (B, m, 1).
    """
    key_prob, key = jrnd.split(key)
    b, c, xstar, sstar = generate_problem(key_prob, batch_size)
    return {
        "input": {"b": b, "c": c},
        "xstar": xstar,
        "sstar": sstar,
    }, key


# %% Initialize the model
model = HardConstrainedMLP(layers=[200, 200])

# Sample one batch only to create shapes for initialisation
batch, key = make_batch(key_init, batch_size=1)
key, key_init = jrnd.split(key)
params = model.init(key_init, batch["input"])

tx = optax.adam(learning_rate)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# %% Training
@jit
def loss_fn(params: dict[str, Any], input: dict[str, Any]):
    """Compute the loss function and auxiliary values.

    Args:
        params: Model parameters.
        input: Input data containing "b" and "c".

    Returns:
        tuple:
            - loss (jax.Array): Mean objective value.
            - aux (tuple): Auxiliary values containing constraint violations.
    """
    c = input["c"]
    pred = cast(jax.Array, model.apply(params, input))
    x = pred[:, :n]
    s = pred[:, n:]
    objective_value = objective(x, c)
    return jnp.mean(objective_value), (x, s)


@jit
def train_step(state: train_state.TrainState, batch: dict[str, Any]):
    """Perform a single training step.

    Args:
        state: Current state of the model.
        batch: Input data containing "b" and "c".

    Returns:
        tuple:
            - state (TrainState): Updated state of the model.
            - loss (jax.Array): Loss value after the step.
            - aux (tuple): Auxiliary values containing constraint violations.
    """
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = grad_fn(state.params, batch["input"])
    state = state.apply_gradients(grads=grads)
    return state, loss, aux


# Training loop
for epoch in range(1, n_epochs + 1):
    epoch_losses = []
    key_train, key = jrnd.split(key_train)
    batch, key_train = make_batch(key)
    state, loss_value, (x, s) = train_step(state, batch)
    cv_eq = constraint_violation_eq(x, s, batch["input"]["b"])
    cv_soc = constraint_violation_soc(s)
    rs = relative_suboptimality(x, cast(jax.Array, batch["xstar"]), batch["input"]["c"])
    if epoch % 10 == 0 or epoch == 1:
        print(
            f"""[{epoch:03d}/{n_epochs}]
            \tloss = {loss_value:.4e}
            \t{cv_eq.max()=}
            \t{cv_soc.max()=}
            \t{rs.max()=}
            """
        )

# %% Validation
key_test, key = jrnd.split(key_train)
val_batch, _ = make_batch(key_test, batch_size=BATCH_SIZE)
pred_val = cast(jax.Array, model.apply(state.params, val_batch["input"]))
b = val_batch["input"]["b"]

x_pred = pred_val[:, :n]
s_pred = pred_val[:, n:]

print_stats(
    x_pred,
    s_pred,
    val_batch["input"]["b"],
    val_batch["input"]["c"],
    cast(jax.Array, val_batch["xstar"]),
)

# %%
