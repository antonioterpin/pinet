"""Generate random second-order cone programming (SOCP) problems."""

import jax.numpy as jnp
from jax import random as jrnd

from benchmarks.toy_SOC.projection_layer import project_soc


def rand_sparse_mask(
    key: jrnd.PRNGKey,
    shape: tuple,
    sparsity: float = 0.01,
    dtype: jnp.dtype = jnp.float64,
):
    """Return a dense tensor whose entries are 0 with prob = `sparsity`.

    Args:
        key (jax.random.PRNGKey): Random key for generating the tensor.
        shape (tuple): Shape of the tensor to be generated.
        sparsity (float): Probability of an entry being zero. Default is 0.01.
        dtype (jnp.dtype): Data type of the tensor. Default is jnp.float64

    Returns:
        jnp.ndarray:
            A tensor of the specified shape with random values and a mask applied.
    """
    key_val, key_mask = jrnd.split(key)

    # Non-zero density is 1 − sparsity
    density = 1.0 - sparsity

    values = jrnd.uniform(key_val, shape, dtype, minval=-1, maxval=1)
    mask = jrnd.bernoulli(key_mask, p=density, shape=shape)
    return values * mask.astype(dtype)


def generate_problem(key: jrnd.PRNGKey, A: jnp.ndarray, B: int):
    """Generate a random linear problem with SOC constraints.

    Args:
        A (jnp.ndarray): Constraint matrix of shape (m, n).
        key (jax.random.PRNGKey): Random key for generating the problem.
        B (int): Number of problem instances to generate.

    Returns:
        tuple:
            - b (jnp.ndarray):
                Right-hand side of the equality constraints, shape (B, m, 1).
            - c (jnp.ndarray): Coefficients for the objective function, shape (B, n, 1).
            - x (jnp.ndarray): Optimal primal solution, shape (B, n, 1).
            - s (jnp.ndarray):
                Optimal dual solution satisfying the SOC constraint, shape (B, m, 1).
    """
    m = A.shape[0]
    n = A.shape[1]
    keyz, keyx = jrnd.split(key)
    z = jrnd.uniform(keyz, (B, m, 1), minval=-1, maxval=1)
    s = project_soc(z)
    y = s - z

    # Generate the primal solution x
    x = jrnd.uniform(keyx, (B, n, 1), minval=-1, maxval=1)
    b = A @ x + s
    c = -A.T @ y

    return b, c, x, s


def objective(x: jnp.ndarray, c: jnp.ndarray):
    """Compute the objective value for the linear problem.

    Args:
        x (jnp.ndarray): Primal solution, shape (B, n, 1).
        c (jnp.ndarray): Coefficients for the objective function, shape (B, n, 1).

    Returns:
        jnp.ndarray: Objective value, shape (B, 1).
    """
    return jnp.sum(c * x, axis=(1, 2), keepdims=True)


def constraint_violation_eq(
    A: jnp.ndarray, x: jnp.ndarray, s: jnp.ndarray, b: jnp.ndarray
):
    """Compute the constraint violation for Ax = b.

    Args:
        A (jnp.ndarray): Constraint matrix, shape (m, n).
        x (jnp.ndarray): Primal solution, shape (B, n, 1).
        s (jnp.ndarray): Dual solution, shape (B, m, 1).
        b (jnp.ndarray): Right-hand side of the equality constraints, shape (B, m, 1).

    Returns:
        jnp.ndarray: Constraint violation, shape (B, 1).
    """
    return jnp.linalg.norm(A @ x + s - b, ord=jnp.inf, axis=-1)


def constraint_violation_soc(s: jnp.ndarray):
    """Compute the constraint violation for the SOC constraint.

    Args:
        s (jnp.ndarray): Dual solution, shape (B, m + 1, 1).

    Returns:
        jnp.ndarray: Constraint violation, shape (B, 1).
    """
    u = s[:, :-1]
    t = s[:, -1:]
    u_norm = jnp.linalg.norm(u, axis=1, keepdims=True)

    return jnp.maximum(u_norm - t, 0.0)


def relative_suboptimality(x: jnp.ndarray, xstar: jnp.ndarray, c: jnp.ndarray):
    """Compute the relative suboptimality of the solution.

    Args:
        x (jnp.ndarray): Primal solution, shape (B, n, 1).
        xstar (jnp.ndarray): Optimal primal solution, shape (B, n, 1).
        c (jnp.ndarray): Coefficients for the objective function, shape (B, n, 1).

    Returns:
        jnp.ndarray: Relative suboptimality, shape (B, 1).
    """
    optimal_val = objective(xstar, c)
    candidate_val = objective(x, c)
    return jnp.abs(candidate_val - optimal_val) / (jnp.abs(optimal_val) + 1e-12)
