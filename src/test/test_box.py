"""Tests for the box constraint."""

import jax.numpy as jnp

from hcnn.constraints.box import BoxConstraint


def test_box():
    lower_bounds = [
        jnp.array([0, 0]),
        jnp.array([0, 0]),
        jnp.array([0, 0]),
        jnp.array([0, 0]),
        jnp.array([-0.5, 0]),
        jnp.array([-0.5, 0]),
    ]
    upper_bounds = [
        jnp.array([1, 1]),
        jnp.array([1, 2]),
        jnp.array([1, 1]),
        jnp.array([1, 1]),
        jnp.array([0.5, 1]),
        jnp.array([0.5, 1]),
    ]
    xs = [
        jnp.array([[2, 2]]),
        jnp.array([[2, 2]]),
        jnp.array([[0.5, 0.5]]),
        jnp.array([[-0.5, -0.5]]),
        jnp.array([[-0.5, 1.5]]),
        jnp.array([[-1.5, 1.5]]),
    ]
    ys = [
        jnp.array([[1, 1]]),
        jnp.array([[1, 2]]),
        jnp.array([[0.5, 0.5]]),
        jnp.array([[0, 0]]),
        jnp.array([[-0.5, 1]]),
        jnp.array([[-0.5, 1]]),
    ]

    for lb, ub, x, y in zip(lower_bounds, upper_bounds, xs, ys):
        box_constraint = BoxConstraint(lb, ub)
        z = box_constraint.project(x)

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
        jnp.array([0]), jnp.array([1]), jnp.array([1, 0], dtype=jnp.bool_)
    )
    x = jnp.array([[2, 2]])

    y = box_constraint.project(x)

    assert y[0, 0] == 1, "The first element should be clipped to 1."
    assert y[0, 1] == 2, "The second element should not be clipped."
