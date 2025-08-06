"""Tests for the box constraint."""

from itertools import product

import jax
import jax.numpy as jnp
import pytest

from pinet import BoxConstraint, ProjectionInstance


def test_instantiation_error():
    try:
        BoxConstraint(jnp.array([0]), jnp.array([1, 2]))
    except Exception:
        pass
    else:
        raise AssertionError("The upper and lower bounds must have the same shape.")

    try:
        BoxConstraint(
            jnp.array([0]), jnp.array([1]), jnp.array([1, 0], dtype=jnp.int32)
        )
    except Exception:
        pass
    else:
        raise AssertionError("The mask must be a boolean array.")

    try:
        BoxConstraint(
            jnp.array([0]), jnp.array([1]), jnp.array([1, 1], dtype=jnp.bool_)
        )
    except Exception:
        pass
    else:
        raise AssertionError("The mask must have the same shape as the bounds.")

    try:
        BoxConstraint(jnp.array([0, 1]), jnp.array([1, 0]))
    except Exception:
        pass
    else:
        raise AssertionError("The lower bound must be less than the upper bound.")


def test_box():
    lower_bounds = [
        jnp.array([0, 0]).reshape((1, 2, 1)),
        jnp.array([0, 0]).reshape((1, 2, 1)),
        jnp.array([0, 0]).reshape((1, 2, 1)),
        jnp.array([0, 0]).reshape((1, 2, 1)),
        jnp.array([-0.5, 0]).reshape((1, 2, 1)),
        jnp.array([-0.5, 0]).reshape((1, 2, 1)),
    ]
    upper_bounds = [
        jnp.array([1, 1]).reshape((1, 2, 1)),
        jnp.array([1, 2]).reshape((1, 2, 1)),
        jnp.array([1, 1]).reshape((1, 2, 1)),
        jnp.array([1, 1]).reshape((1, 2, 1)),
        jnp.array([0.5, 1]).reshape((1, 2, 1)),
        jnp.array([0.5, 1]).reshape((1, 2, 1)),
    ]
    xs = [
        jnp.array([[2, 2]]).reshape((1, 2, 1)),
        jnp.array([[2, 2]]).reshape((1, 2, 1)),
        jnp.array([[0.5, 0.5]]).reshape((1, 2, 1)),
        jnp.array([[-0.5, -0.5]]).reshape((1, 2, 1)),
        jnp.array([[-0.5, 1.5]]).reshape((1, 2, 1)),
        jnp.array([[-1.5, 1.5]]).reshape((1, 2, 1)),
    ]
    ys = [
        jnp.array([[1, 1]]).reshape((1, 2, 1)),
        jnp.array([[1, 2]]).reshape((1, 2, 1)),
        jnp.array([[0.5, 0.5]]).reshape((1, 2, 1)),
        jnp.array([[0, 0]]).reshape((1, 2, 1)),
        jnp.array([[-0.5, 1]]).reshape((1, 2, 1)),
        jnp.array([[-0.5, 1]]).reshape((1, 2, 1)),
    ]

    for lb, ub, x, y in zip(lower_bounds, upper_bounds, xs, ys):
        box_constraint = BoxConstraint(lb, ub)
        z = box_constraint.project(ProjectionInstance(x=x))

        assert jnp.allclose(
            y, z
        ), f"""
            Projection of {x} onto:
                lb: {lb}
                ub: {ub}
            should be {y}, instead of {z}.
        """


def test_mask():
    box_constraint = BoxConstraint(
        jnp.array([0]).reshape((1, 1, 1)),
        jnp.array([1]).reshape((1, 1, 1)),
        jnp.array([1, 0], dtype=jnp.bool_),
    )
    x = jnp.array([[2, 2]]).reshape((1, 2, 1))
    y = box_constraint.project(ProjectionInstance(x=x))

    assert y[0, 0, 0] == 1, "The first element should be clipped to 1."
    assert y[0, 1, 0] == 2, "The second element should not be clipped."


DIM = 100
N_BATCH = [1, 10]
SEED = [24, 42]


@pytest.mark.parametrize(
    "n_batch_l, n_batch_u, n_batch_x, seed", product(N_BATCH, N_BATCH, N_BATCH, SEED)
)
def test_box_parametrized(n_batch_l, n_batch_u, n_batch_x, seed):
    if n_batch_l > n_batch_x or n_batch_u > n_batch_x:
        return
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=4)
    mask = jax.random.bernoulli(key[0], shape=(DIM)).astype(jnp.bool_)
    active_entries = mask.sum().item()
    lb = jax.random.uniform(
        key[1], shape=(n_batch_l, active_entries, 1), minval=0, maxval=1
    )
    ub = jax.random.uniform(
        key[2], shape=(n_batch_u, active_entries, 1), minval=lb.max(axis=0), maxval=1
    )
    x = jax.random.uniform(key[3], shape=(n_batch_x, DIM, 1), minval=-2, maxval=2)

    box_constraint = BoxConstraint(lb, ub, mask)
    z = box_constraint.project(ProjectionInstance(x=x))

    # Compute projectino with for loop
    y = x.copy()
    for ii in range(n_batch_x):
        y = y.at[ii, mask, :].set(
            jnp.clip(
                x[ii, mask, :],
                min=lb[min(ii, n_batch_l), :, :],
                max=ub[min(ii, n_batch_u), :, :],
            )
        )

    assert jnp.allclose(y, z), "Projection should be the same as the for loop."
