"""Tests for the equality constraint."""

import re
from itertools import product
from typing import cast

import cvxpy as cp
import jax
import jax.numpy as jnp
import pytest
from cvxpy.constraints.constraint import Constraint as CvxConstraint

from pinet import EqualityConstraint, ProjectionInstance

# Set JAX precision to 64 bits.
jax.config.update("jax_enable_x64", True)


# Test Parameters
# Vector dimension
DIM = 100
# Random seeds
SEEDS = [0, 24]
# Methods for equality constraints (paired with above seeds)
VALID_METHODS = ["pinv"]
# Number of equality constraints (for QP test)
N_EQ = 50
# Batch size options
N_BATCH = [1, 10]


def test_instantiation_error():
    try:
        EqualityConstraint(jnp.array([1]), jnp.array([[[1]]]))
    except Exception:
        pass
    else:
        raise AssertionError(
            "No check that a is a matrix with shape (n_batch, n_constraints, dimension)."
        )

    try:
        EqualityConstraint(jnp.array([[[1]]]), jnp.array([[1]]))
    except Exception:
        pass
    else:
        raise AssertionError(
            "No check that b must have shape (n_batch, n_constraints, 1)."
        )

    try:
        EqualityConstraint(jnp.array([[[1]]]), jnp.array([[[1, 2]]]))
    except Exception:
        pass
    else:
        raise AssertionError(
            "No check that b must have shape (n_batch, n_constraints, 1)."
        )

    try:
        EqualityConstraint(jnp.array([[[1]], [[1]], [[1]]]), jnp.array([[[1]], [[1]]]))
    except Exception:
        pass
    else:
        raise AssertionError("No check that batch sizes are consistent.")

    try:
        EqualityConstraint(jnp.array([[[1], [1], [1]]]), jnp.array([[[1]], [[1]]]))
    except Exception:
        pass
    else:
        raise AssertionError("Number of rows in a must equal size of b.")


@pytest.mark.parametrize(
    "method, seed, n_batch_a, n_batch_b, n_batch_x",
    product(VALID_METHODS, SEEDS, N_BATCH, N_BATCH, N_BATCH),
)
def test_equality_eye(method, seed, n_batch_a, n_batch_b, n_batch_x):
    # a and b always have smaller or equal batch size than that of x.
    if n_batch_a > n_batch_x or n_batch_b > n_batch_x:
        return
    # Instantiate key
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=2)
    # Equality constraint
    a = jnp.repeat(jnp.expand_dims(jnp.eye(DIM), axis=0), n_batch_a, axis=0)
    b = jax.random.uniform(key[0], shape=(n_batch_b, DIM, 1))
    # Point to be projected. Shape (n_batch, dimension, 1).
    x = jax.random.uniform(key[1], shape=(n_batch_x, DIM, 1))
    # True projection
    y = b

    # Instantiate object and project
    eq_constraint = EqualityConstraint(a, b, method=method)
    # Prepare input
    inp = ProjectionInstance(x=x)
    z = eq_constraint.project(inp).x
    assert jnp.allclose(z, y), (
        "Projection onto the identity equality constraint should equal b. "
        f"Expected {y}, got {z}."
    )


@pytest.mark.parametrize(
    "method, seed, n_batch_a, n_batch_b, n_batch_x",
    product(VALID_METHODS, SEEDS, N_BATCH, N_BATCH, N_BATCH),
)
def test_equality_diagonal(method, seed, n_batch_a, n_batch_b, n_batch_x):
    # a and b always have smaller or equal batch size than that of x.
    if n_batch_a > n_batch_x or n_batch_b > n_batch_x:
        return
    # Instantiate key
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=2)
    # Test with invertible diagonal matrix
    a = jnp.repeat(
        jnp.expand_dims(jnp.diag(jnp.arange(DIM) + 1), axis=0), n_batch_a, axis=0
    )
    b = jax.random.uniform(key[0], shape=(n_batch_b, DIM, 1))
    x = jax.random.uniform(key[1], shape=(n_batch_x, DIM, 1))
    # True projection
    y = jnp.linalg.inv(a) @ b

    # Instantiate object and project
    eq_constraint = EqualityConstraint(a, b, method=method)
    # Prepare input
    inp = ProjectionInstance(x=x)
    z = eq_constraint.project(inp).x
    assert jnp.allclose(z, y), (
        "Projection onto an invertible diagonal equality constraint should match "
        f"the analytical solution. Expected {y}, got {z}."
    )


@pytest.mark.parametrize(
    "method, seed, n_batch_a, n_batch_b, n_batch_x",
    product(VALID_METHODS, SEEDS, N_BATCH, N_BATCH, N_BATCH),
)
def test_equality_generic_invertible(method, seed, n_batch_a, n_batch_b, n_batch_x):
    # a and b always have smaller or equal batch size than that of x.
    if n_batch_a > n_batch_x or n_batch_b > n_batch_x:
        return
    # Test with generic invertible matrix
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=3)
    # Generate random matrix
    a = jax.random.uniform(key[0], shape=(n_batch_a, DIM, DIM))
    # Normalize
    a = (a @ jnp.matrix_transpose(a) + jnp.eye(DIM)) + (a - jnp.matrix_transpose(a))
    b = jax.random.uniform(key[1], shape=(n_batch_b, DIM, 1))
    x = jax.random.uniform(key[2], shape=(n_batch_x, DIM, 1))
    # True projection is a^{-1} @ b (a is invertible)
    y = jnp.linalg.solve(a, b)

    # Instantiate object and project
    eq_constraint = EqualityConstraint(a, b, method=method)
    # Prepare input
    inp = ProjectionInstance(x=x)
    z = eq_constraint.project(inp).x
    assert jnp.allclose(z, y), (
        "Projection onto a generic invertible equality constraint should match "
        f"the linear solve reference. Expected {y}, got {z}."
    )


@pytest.mark.parametrize(
    "method, seed, n_batch_a, n_batch_b, n_batch_x",
    product(VALID_METHODS, SEEDS, N_BATCH, N_BATCH, N_BATCH),
)
def test_equality_qp(method, seed, n_batch_a, n_batch_b, n_batch_x):
    # a and b always have smaller or equal batch size than that of x.
    if n_batch_a > n_batch_x or n_batch_b > n_batch_x:
        return
    # Test with generic matrix
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=3)
    a = jax.random.uniform(key[0], shape=(n_batch_a, N_EQ, DIM))
    # Generate RHS b vector
    if n_batch_a > n_batch_b:
        # If we have more a than b, then we need to find
        # an element in the intersection of their ranges.
        x_list, constraints = [], []
        b_common = cp.Variable(N_EQ)
        for ii in range(n_batch_a):
            x_list.append(cp.Variable(DIM))
            constraints += [a[ii, :, :] @ x_list[ii] == b_common]
        # Add this constraint to keep the solution bounded
        constraints += [-1 <= x_list[0], x_list[0] <= 1]
        # Add some objective (not important), to avoid the trivial solution
        objective = cp.Minimize(jnp.ones(shape=(DIM,)) @ x_list[0])
        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve(verbose=False)
        b = jnp.reshape(jnp.array(b_common.value), shape=(n_batch_b, N_EQ, 1))
    else:
        # If we have more b than a, then we generate b in the range of a
        b = a @ jax.random.uniform(key[1], shape=(n_batch_b, DIM, 1))

    x = jax.random.uniform(key[2], shape=(n_batch_x, DIM, 1))
    # Vectors should have shape (dimension,). If (dimension,1) it crashes
    # Instantiate object and project
    eq_constraint = EqualityConstraint(a, b, method=method)
    inp = ProjectionInstance(x=x)
    z = eq_constraint.project(inp).x
    # Compute true projection by solving equality constrained QP
    for ii in range(n_batch_x):
        ycp = cp.Variable(DIM)
        objective = cp.Minimize(cp.sum_squares(x[ii, :, 0] - ycp))
        constraints = cast(
            list[CvxConstraint],
            [a[min(ii, n_batch_a - 1), :, :] @ ycp == b[min(ii, n_batch_b - 1), :, 0]],
        )
        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve()
        # Extract true projection
        y = jnp.expand_dims(jnp.array(ycp.value), axis=1)

        assert jnp.allclose(z[ii, :], y), (
            "Equality projection should match the CVXPY solution for each batch "
            f"element. Batch {ii}: expected {y}, got {z[ii, :]}."
        )


def test_equality_method_none_raises_not_implemented():
    # Minimal valid shapes
    a = jnp.expand_dims(jnp.eye(1), axis=0)  # (1, 1, 1)
    b = jnp.zeros((1, 1, 1))
    eq_constraint = EqualityConstraint(a, b, method=None)

    # Calling project should raise NotImplementedError
    inp = ProjectionInstance(x=jnp.zeros((1, 1, 1)))
    with pytest.raises(NotImplementedError, match=re.escape("No projection method set.")):
        eq_constraint.project(inp)


def test_equality_invalid_method_raises_exception():
    a = jnp.expand_dims(jnp.eye(1), axis=0)  # (1, 1, 1)
    b = jnp.zeros((1, 1, 1))
    with pytest.raises(
        Exception, match=r"Invalid method foo\. Valid methods are: \['pinv', None\]"
    ):
        EqualityConstraint(a, b, method="foo")
