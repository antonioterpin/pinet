"""Tests for the second-order cone (SOC) constraint."""

from itertools import product
from typing import cast

import cvxpy as cp
import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np
import pytest
from cvxpy.constraints.constraint import Constraint as CvxpyConstraint

from pinet import (
    NonLinearSpecification,
    ProjectionInstance,
    SocConstraint,
    SocConstraintSpecification,
    SOCType,
)
from pinet.constraints.non_linear import NonLinearConstraint

jax.config.update("jax_enable_x64", True)

# 2D tests (point-to-project, projection)
twod_tests = [
    # Tests that map to zero
    ([0.0, 0.0], [0.0, 0.0]),
    ([0.0, -1.0], [0.0, 0.0]),
    ([-0.5, -1], [0.0, 0.0]),
    ([0.75, -2], [0.0, 0.0]),
    # Tests that map to themselves
    ([0.5, 1.0], [0.5, 1.0]),
    ([2.5, 4.0], [2.5, 4.0]),
    # That map to cone boundary
    ([3.0, 1.0], [2.0, 2.0]),
    ([5.0, 4.0], [4.5, 4.5]),
    ([-3.0, 1.0], [-2.0, 2.0]),
    ([-5.0, 4.0], [-4.5, 4.5]),
]

# Dimension of larger tests
DIM = 100


@pytest.mark.parametrize("x, y", twod_tests)
def test_soc(x, y):
    x = jnp.array(x).reshape((1, 2, 1))
    y = jnp.array(y).reshape((1, 2, 1))

    mask_u = jnp.array([1, 0], dtype=jnp.bool_)
    mask_t = jnp.array([0, 1], dtype=jnp.bool_)
    socspec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t)
    soc_constraint = SocConstraint(socspec=socspec)
    z = soc_constraint.project(ProjectionInstance(x=x)).x

    assert jnp.allclose(y, z), f"""
        Projection of {x} onto SOC should be {y}, instead of {z}.
    """


@pytest.mark.parametrize("x, y", twod_tests)
def test_mask(x, y):
    x = jnp.array(x).reshape((1, 2, 1))
    y = jnp.array(y).reshape((1, 2, 1))
    seed = 0
    key = jrnd.PRNGKey(seed)
    key = jrnd.split(key, num=2)
    # Increase dimension of x and y
    x = jnp.concatenate((x, jrnd.uniform(key[0], (1, DIM - 2, 1))), axis=1)
    y = jnp.concatenate((y, jrnd.uniform(key[1], (1, DIM - 2, 1))), axis=1)
    # Define masks
    mask_u = jnp.zeros((DIM), dtype=jnp.bool_).at[0].set(True)
    mask_t = jnp.zeros((DIM), dtype=jnp.bool_).at[1].set(True)
    socspec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t)
    soc_constraint = SocConstraint(socspec=socspec)
    z = soc_constraint.project(ProjectionInstance(x=x)).x

    assert jnp.allclose(y[:, :2, :], z[:, :2, :]), f"""
            Projection of {x} onto SOC should be {y}, instead of {z}.
        """


SEEDS = [24, 42]
BATCH_SIZES = [1, 10]


@pytest.mark.parametrize(
    "seed, batch_size",
    product(SEEDS, BATCH_SIZES),
)
def test_soc_a_b_parametrized(seed, batch_size):
    key = jrnd.PRNGKey(seed)
    key_points, key = jrnd.split(key)
    x = jrnd.uniform(key=key_points, shape=(batch_size, DIM, 1), minval=-10, maxval=10)
    key_active, key = jrnd.split(key)
    # Only the first active_dims will be constraint in the SOC
    active_dims = jrnd.randint(key=key_active, shape=(), minval=1, maxval=DIM - 1).item()
    # Define projection with cvxpy
    y_cvxpy = cp.Variable(DIM)
    x_cvxpy = cp.Parameter(DIM)
    a_cvxpy = cp.Parameter(active_dims)
    b_cvxpy = cp.Parameter(1)
    constraints = [
        cp.SOC(y_cvxpy[active_dims] + b_cvxpy, y_cvxpy[:active_dims] + a_cvxpy)
    ]
    objective = cp.Minimize(cp.sum_squares(y_cvxpy - x_cvxpy))
    problem = cp.Problem(
        objective=objective, constraints=cast(list[CvxpyConstraint], constraints)
    )

    ysocp = jnp.zeros((batch_size, DIM, 1))
    for ii in range(batch_size):
        x_cvxpy.value = np.array(x[ii, :, 0])
        a_cvxpy.value = np.zeros(active_dims)
        b_cvxpy.value = np.zeros(1)
        problem.solve(solver=cp.SCS, eps_abs=1e-10, eps_rel=1e-10, verbose=False)
        ysocp = ysocp.at[ii, :, 0].set(y_cvxpy.value)

    mask_u = jnp.zeros((DIM), dtype=jnp.bool_).at[:active_dims].set(True)
    mask_t = jnp.zeros((DIM), dtype=jnp.bool_).at[active_dims].set(True)
    socspec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t)
    soc_constraint = SocConstraint(socspec=socspec)
    project_soc = jax.jit(lambda y_raw: soc_constraint.project(y_raw=y_raw).x)
    z = project_soc(ProjectionInstance(x=x))

    assert jnp.allclose(ysocp, z)
    # Generate random affine terms
    key_a, key_b = jrnd.split(key, num=2)
    a = jrnd.uniform(key=key_a, shape=(batch_size, active_dims, 1))
    b = jrnd.uniform(key=key_b, shape=(batch_size, 1, 1), minval=0)
    ysocp_aff = jnp.zeros((batch_size, DIM, 1))
    for ii in range(batch_size):
        x_cvxpy.value = np.array(x[ii, :, 0])
        a_cvxpy.value = np.array(a[ii, :, 0])
        b_cvxpy.value = np.array(b[ii, :, 0])
        problem.solve(solver=cp.SCS, eps_abs=1e-10, eps_rel=1e-10, verbose=False)
        ysocp_aff = ysocp_aff.at[ii, :, 0].set(y_cvxpy.value)

    socspec_aff = SocConstraintSpecification(a=a, b=b)
    z_aff = project_soc(ProjectionInstance(x=x, soc=socspec_aff))
    assert jnp.allclose(ysocp_aff, z_aff)

    socspec_updated = socspec.update(a=a, b=b, mask_u=None, mask_t=None)
    z_updated = project_soc(ProjectionInstance(x=x, soc=socspec_updated))
    assert jnp.allclose(ysocp_aff, z_updated)


def test_raising_errors():
    with pytest.raises(ValueError):
        SocConstraint(socspec=SocConstraintSpecification(mask_u=None, mask_t=None))

    mask_u = jnp.ones((DIM), dtype=jnp.bool_).at[-1].set(False)
    mask_t = jnp.zeros((DIM), dtype=jnp.bool_).at[-1].set(True)
    socspec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t)
    x = jnp.ones((1, DIM, 1))
    soc_constraint = SocConstraint(socspec=socspec)

    with pytest.raises(ValueError):
        soc_constraint.project(ProjectionInstance(x=x, soc=socspec))


def test_nl_type_property_reports_soc():
    """``NonLinearConstraint.nl_type`` exposes the carried non-linear type."""
    spec = NonLinearSpecification(nl_type=SOCType, a_mat=jnp.ones((1, 2, 3)))
    constraint = NonLinearConstraint(spec=spec)
    assert constraint.nl_type == SOCType
