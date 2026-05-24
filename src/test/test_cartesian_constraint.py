"""Tests for the CartesianConstraint class."""

from itertools import product
from typing import cast

import cvxpy as cp
import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np
import pytest
from cvxpy.constraints.constraint import Constraint as CvxpyConstraint
from jaxtyping import TypeCheckError

from pinet import (
    BoxConstraint,
    BoxConstraintSpecification,
    CartesianConstraint,
    NonLinearSpecification,
    ProjectionInstance,
    SocConstraint,
    SocConstraintSpecification,
)
from pinet.constraints.base import Constraint

# Beartype (when ``PINET_RUNTIME_CHECK=1``) pre-empts the library's own
# type/value checks at ``__init__`` time, so negative-path tests accept
# either ``TypeCheckError`` or the narrower library error.
_TypeOrValidationError = (TypeError, ValueError, TypeCheckError)

jax.config.update("jax_enable_x64", True)

DIM = 100

SEEDS = [24, 42]
BATCH_SIZES = [1, 10]


def test_empty_input_raises():
    """Test that empty constraint list raises ValueError."""
    with pytest.raises(ValueError, match=r"At least one constraint must be provided"):
        CartesianConstraint(box_constraint=None, nl_constraints=[])

    with pytest.raises(ValueError, match=r"At least one constraint must be provided"):
        CartesianConstraint(box_constraint=None, nl_constraints=None)


def test_nl_constraints_is_iterable():
    """Test that non-iterable nl_constraints raises TypeError."""
    with pytest.raises(_TypeOrValidationError):
        # Deliberately wrong type to exercise the input-validation error path.
        CartesianConstraint(nl_constraints=cast(list[SocConstraint], cast(object, 5)))


def test_non_box_soc_constraint_raises():
    """Test that non-Box/SOC constraint types raise ValueError."""

    class DummyConstraint(Constraint):
        _dim: int = 5

        def project(self, y_raw):
            return y_raw

        def cv(self, y_raw):
            return jnp.zeros((1, 1, 1))

        @property
        def dim(self):
            return self._dim

        @property
        def n_constraints(self):
            return 1

    dummy = DummyConstraint()

    with pytest.raises(_TypeOrValidationError):
        # Deliberately wrong type to exercise the input-validation error path.
        CartesianConstraint(box_constraint=cast(BoxConstraint, cast(object, dummy)))

    with pytest.raises(_TypeOrValidationError):
        # Deliberately wrong type to exercise the input-validation error path.
        CartesianConstraint(nl_constraints=cast(list[SocConstraint], [dummy]))


def test_project_raises_when_nl_specs_not_iterable():
    """Test project raises TypeError if y_raw.nl is not a list/tuple."""
    dim = 6
    soc_mask_u = jnp.array([True, True, False, False, False, False], dtype=jnp.bool_)
    soc_mask_t = jnp.array([False, False, True, False, False, False], dtype=jnp.bool_)
    soc = SocConstraint(SocConstraintSpecification(mask_u=soc_mask_u, mask_t=soc_mask_t))
    cartesian = CartesianConstraint(nl_constraints=[soc])

    # y_raw.nl must be list/tuple when nonlinear constraints are present.
    # ProjectionInstance's own __init__ is beartype-checked; constructing the
    # deliberately-malformed instance may raise a TypeCheckError before
    # project() runs. Either error path is acceptable.
    with pytest.raises(_TypeOrValidationError):
        y_raw = ProjectionInstance(
            x=jnp.zeros((1, dim, 1)),
            nl=cast(
                "list[NonLinearSpecification]",
                cast(
                    object,
                    SocConstraintSpecification(mask_u=soc_mask_u, mask_t=soc_mask_t),
                ),
            ),
        )
        cartesian.project(y_raw)


def test_wrong_dimensions_box_and_soc_raise():
    """Test that Box and SOC constraints with different dimensions raise ValueError."""
    # Create box constraint with dimension 5
    mask_box = jnp.array([True, True, False, False, False], dtype=jnp.bool_)
    box = BoxConstraint(
        BoxConstraintSpecification(
            lb=jnp.array([[[-1.0], [-1.0]]]),
            ub=jnp.array([[[1.0], [1.0]]]),
            mask=mask_box,
        )
    )

    # Create SOC constraint with dimension 10
    soc_mask_u = jnp.array([True] * 3 + [False] * 7, dtype=jnp.bool_)
    soc_mask_t = jnp.array([False] * 3 + [True] + [False] * 6, dtype=jnp.bool_)
    soc = SocConstraint(SocConstraintSpecification(mask_u=soc_mask_u, mask_t=soc_mask_t))

    with pytest.raises(ValueError, match=r"All constraints must have the same dimension"):
        CartesianConstraint(box_constraint=box, nl_constraints=[soc])


def test_overlapping_box_and_soc_masks_raise():
    """Test that overlapping masks between Box and SOC constraints raise ValueError."""
    # Create box constraint on dimensions 0-1
    mask_box = jnp.array([True, True, False, False, False], dtype=jnp.bool_)
    box = BoxConstraint(
        BoxConstraintSpecification(
            lb=jnp.array([[[-1.0], [-1.0]]]),
            ub=jnp.array([[[1.0], [1.0]]]),
            mask=mask_box,
        )
    )

    # Create SOC constraint that overlaps (dimension 1-2)
    mask_u = jnp.array([False, True, False, False, False], dtype=jnp.bool_)
    mask_t = jnp.array([False, False, True, False, False], dtype=jnp.bool_)
    soc = SocConstraint(SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t))

    with pytest.raises(ValueError, match=r"Constraint masks overlap"):
        CartesianConstraint(box_constraint=box, nl_constraints=[soc])


def test_wrong_dimensions_two_soc_constraints_raise():
    """Test that SOC constraints with different dimensions raise ValueError."""
    soc_1 = SocConstraint(
        SocConstraintSpecification(
            mask_u=jnp.array([True, True, False, False, False], dtype=jnp.bool_),
            mask_t=jnp.array([False, False, True, False, False], dtype=jnp.bool_),
        )
    )
    soc_2 = SocConstraint(
        SocConstraintSpecification(
            mask_u=jnp.array([True] * 4 + [False] * 4, dtype=jnp.bool_),
            mask_t=jnp.array([False] * 4 + [True] + [False] * 3, dtype=jnp.bool_),
        )
    )

    with pytest.raises(
        ValueError, match=r"All constraints must have the same dimension."
    ):
        CartesianConstraint(nl_constraints=[soc_1, soc_2])


def test_overlapping_two_soc_masks_raise():
    """Test that overlapping masks between two SOC constraints raise ValueError."""
    soc_1 = SocConstraint(
        SocConstraintSpecification(
            mask_u=jnp.array([True, False, False, False, False], dtype=jnp.bool_),
            mask_t=jnp.array([False, True, False, False, False], dtype=jnp.bool_),
        )
    )
    soc_2 = SocConstraint(
        SocConstraintSpecification(
            mask_u=jnp.array([False, True, False, False, False], dtype=jnp.bool_),
            mask_t=jnp.array([False, False, True, False, False], dtype=jnp.bool_),
        )
    )

    with pytest.raises(
        ValueError,
        match=r"Constraint masks overlap with previously defined constraints.",
    ):
        CartesianConstraint(nl_constraints=[soc_1, soc_2])


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZES))
def test_random_example_with_two_boxes_two_socs(seed: int, batch_size: int):
    """Test a random example with 1 box and 2 SOCs in 100 dimensions."""
    key = jrnd.PRNGKey(seed)
    box_1_bound = 2.0
    box_2_bound = 1.0

    # Create non-overlapping masks for 100 dimensions
    # Box 1: dimensions 0-24
    mask_box_1 = jnp.zeros(DIM, dtype=jnp.bool_)
    mask_box_1 = mask_box_1.at[0:25].set(True)

    # Box 2: dimensions 25-49
    mask_box_2 = jnp.zeros(DIM, dtype=jnp.bool_)
    mask_box_2 = mask_box_2.at[25:50].set(True)

    # SOC 1: dimensions 50-74 (u) and 75 (t)
    soc_mask_u_1 = jnp.zeros(DIM, dtype=jnp.bool_)
    soc_mask_u_1 = soc_mask_u_1.at[50:75].set(True)
    soc_mask_t_1 = jnp.zeros(DIM, dtype=jnp.bool_)
    soc_mask_t_1 = soc_mask_t_1.at[75].set(True)

    # SOC 2: dimensions 76-89 (u) and 90 (t)
    soc_mask_u_2 = jnp.zeros(DIM, dtype=jnp.bool_)
    soc_mask_u_2 = soc_mask_u_2.at[76:90].set(True)
    soc_mask_t_2 = jnp.zeros(DIM, dtype=jnp.bool_)
    soc_mask_t_2 = soc_mask_t_2.at[90].set(True)

    # Create box constraints
    box = BoxConstraint(
        BoxConstraintSpecification(
            lb=jnp.array([-box_1_bound] * 25 + [-box_2_bound] * 25).reshape(1, -1, 1),
            ub=jnp.array([box_1_bound] * 25 + [box_2_bound] * 25).reshape(1, -1, 1),
            mask=jnp.logical_or(mask_box_1, mask_box_2),
        )
    )

    # Create SOC constraints
    socspec1 = SocConstraintSpecification(mask_u=soc_mask_u_1, mask_t=soc_mask_t_1)
    soc_1 = SocConstraint(socspec=socspec1)

    socspec2 = SocConstraintSpecification(mask_u=soc_mask_u_2, mask_t=soc_mask_t_2)
    soc_2 = SocConstraint(socspec=socspec2)

    # Create CartesianConstraint
    cartesian = CartesianConstraint(box_constraint=box, nl_constraints=[soc_1, soc_2])

    # Verify properties
    assert cartesian.dim == DIM
    # box_1 has 25 constraints, box_2 has 25 constraints, soc_1 has 1, soc_2 has 1
    assert cartesian.n_constraints == 25 + 25 + 1 + 1

    # Generate random points
    key, subkey = jrnd.split(key)
    x = jrnd.uniform(subkey, shape=(batch_size, DIM, 1), minval=-5, maxval=5)
    projection_instance = ProjectionInstance(
        x=x, nl=[socspec1.to_nl_spec(), socspec2.to_nl_spec()]
    )

    # Project using CartesianConstraint
    cart_jit = jax.jit(lambda projinst: cartesian.project(projinst))
    projected = cart_jit(projection_instance)

    # Verify projection properties
    # Box 1 constraints should be satisfied
    assert jnp.all(projected.x[:, 0:25, :] >= -box_1_bound)
    assert jnp.all(projected.x[:, 0:25, :] <= box_1_bound)

    # Box 2 constraints should be satisfied
    assert jnp.all(projected.x[:, 25:50, :] >= -box_2_bound)
    assert jnp.all(projected.x[:, 25:50, :] <= box_2_bound)

    # SOC 1 constraint should be satisfied
    norm_u_1 = jnp.linalg.norm(projected.x[:, 50:75, :], axis=1, keepdims=True)
    t_1 = projected.x[:, 75:76, :]
    assert jnp.all(norm_u_1 <= t_1 + 1e-10)

    # SOC 2 constraint should be satisfied
    norm_u_2 = jnp.linalg.norm(projected.x[:, 76:90, :], axis=1, keepdims=True)
    t_2 = projected.x[:, 90:91, :]
    assert jnp.all(norm_u_2 <= t_2 + 1e-10)  # Small tolerance for numerical errors

    # Dimensions not covered by any constraint should remain unchanged (91-99)
    assert jnp.allclose(projected.x[:, 91:, :], x[:, 91:, :])

    # Verify constraint violation
    cv_before = cartesian.cv(projection_instance)
    cv_after = cartesian.cv(projected)

    # CV after projection should be near zero
    assert jnp.all(cv_after < 1e-6)

    # CV should be a valid array with correct shape
    assert cv_before.shape == (batch_size, 1, 1)
    assert cv_after.shape == (batch_size, 1, 1)

    # Compute exact projection with CVXPY
    y_cvxpy = cp.Variable(DIM)
    x_cvxpy = cp.Parameter(DIM)
    constraints = [
        -box_1_bound <= y_cvxpy[:25],
        y_cvxpy[:25] <= box_1_bound,
        -box_2_bound <= y_cvxpy[25:50],
        y_cvxpy[25:50] <= box_2_bound,
        cp.SOC(y_cvxpy[75], y_cvxpy[50:75]),
        cp.SOC(y_cvxpy[90], y_cvxpy[76:90]),
    ]
    objective = cp.Minimize(cp.sum_squares(y_cvxpy - x_cvxpy))
    problem = cp.Problem(
        objective=objective, constraints=cast(list[CvxpyConstraint], constraints)
    )

    y_exact = jnp.zeros((batch_size, DIM, 1))
    for ii in range(batch_size):
        x_cvxpy.value = np.array(x[ii, :, 0])
        problem.solve(solver=cp.SCS, eps_abs=1e-10, eps_rel=1e-10, verbose=False)
        y_exact = y_exact.at[ii, :, 0].set(y_cvxpy.value)

    assert jnp.allclose(projected.x, y_exact, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZES))
def test_projection_equivalence(seed, batch_size):
    """Test that CartesianConstraint projection equals individual projections."""
    key = jrnd.PRNGKey(seed)
    dim = 10

    # Create non-overlapping constraints
    mask_box = jnp.array([True] * 4 + [False] * 6, dtype=jnp.bool_)
    box = BoxConstraint(
        BoxConstraintSpecification(
            lb=jnp.array([[[-1.0]] * 4]),
            ub=jnp.array([[[1.0]] * 4]),
            mask=mask_box,
        )
    )

    soc_mask_u = jnp.array([False] * 4 + [True] * 3 + [False] * 3, dtype=jnp.bool_)
    soc_mask_t = jnp.array([False] * 7 + [True] + [False] * 2, dtype=jnp.bool_)
    socspec = SocConstraintSpecification(mask_u=soc_mask_u, mask_t=soc_mask_t)
    soc = SocConstraint(
        socspec=socspec,
    )

    # Create CartesianConstraint
    cartesian = CartesianConstraint(box_constraint=box, nl_constraints=[soc])

    # Generate random points
    key, subkey = jrnd.split(key)
    x = jrnd.uniform(subkey, shape=(batch_size, dim, 1), minval=-3, maxval=3)
    projection_instance = ProjectionInstance(x=x, nl=[socspec.to_nl_spec()])

    # Project using CartesianConstraint
    cart_jit = jax.jit(lambda projinst: cartesian.project(projinst))
    cartesian_proj = cart_jit(projection_instance)

    # Project using individual constraints
    box_proj = box.project(projection_instance)
    soc_proj = soc.project(box_proj)

    # Results should be identical
    assert jnp.allclose(cartesian_proj.x, soc_proj.x)


def test_cv():
    """Test that cv returns the maximum violation across all constraints."""
    # Box constraint on dimensions 0-2: [-1, 1]
    mask_box = jnp.array([True] * 3 + [False] * 7, dtype=jnp.bool_)
    box = BoxConstraint(
        BoxConstraintSpecification(
            lb=jnp.array([[[-1.0]] * 3]),
            ub=jnp.array([[[1.0]] * 3]),
            mask=mask_box,
        )
    )

    # SOC constraint on dimensions 3-4
    soc_mask_u = jnp.array([False] * 3 + [True] + [False] * 6, dtype=jnp.bool_)
    soc_mask_t = jnp.array([False] * 4 + [True] + [False] * 5, dtype=jnp.bool_)
    socspec = SocConstraintSpecification(mask_u=soc_mask_u, mask_t=soc_mask_t)
    soc = SocConstraint(
        socspec=socspec,
    )

    cartesian = CartesianConstraint(box_constraint=box, nl_constraints=[soc])

    # The box violation is 4.0 and SOC violation is 0.5
    x = jnp.array([[5.0, 0.0, 0.0, 2.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).reshape(1, 10, 1)
    projection_instance = ProjectionInstance(x=x, nl=[socspec.to_nl_spec()])

    cv_cartesian = cartesian.cv(projection_instance)
    cv_box = box.cv(projection_instance)
    cv_soc = soc.cv(projection_instance)

    # CartesianConstraint cv should equal the maximum of individual cvs
    expected_max = jnp.maximum(cv_box, cv_soc)
    assert jnp.allclose(cv_cartesian, expected_max)


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZES))
def test_projection_with_two_socs_no_box(seed: int, batch_size: int):
    """Test projection with two SOC constraints and no box constraints."""
    key = jrnd.PRNGKey(seed)
    dim = 8

    # SOC 1 uses dims [0, 1] as u and dim [2] as t.
    soc1_mask_u = jnp.array([True, True, False, False, False, False, False, False])
    soc1_mask_t = jnp.array([False, False, True, False, False, False, False, False])
    socspec1 = SocConstraintSpecification(mask_u=soc1_mask_u, mask_t=soc1_mask_t)
    soc1 = SocConstraint(socspec=socspec1)

    # SOC 2 uses dims [3, 4, 5] as u and dim [6] as t.
    soc2_mask_u = jnp.array([False, False, False, True, True, True, False, False])
    soc2_mask_t = jnp.array([False, False, False, False, False, False, True, False])
    socspec2 = SocConstraintSpecification(mask_u=soc2_mask_u, mask_t=soc2_mask_t)
    soc2 = SocConstraint(socspec=socspec2)

    cartesian = CartesianConstraint(nl_constraints=[soc1, soc2])

    key, subkey = jrnd.split(key)
    x = jrnd.uniform(subkey, shape=(batch_size, dim, 1), minval=-4, maxval=4)
    projection_instance = ProjectionInstance(
        x=x,
        nl=[socspec1.to_nl_spec(), socspec2.to_nl_spec()],
    )

    cartesian_proj = cartesian.project(projection_instance)

    # Compare against sequential application of the two SOC projections.
    seq_proj = soc1.project(projection_instance)
    seq_proj = soc2.project(seq_proj)

    assert jnp.allclose(cartesian_proj.x, seq_proj.x)

    # Unconstrained dimension should remain unchanged.
    assert jnp.allclose(cartesian_proj.x[:, 7:8, :], x[:, 7:8, :])

    # After projection, violation should be numerically near zero.
    cv_after = cartesian.cv(cartesian_proj)
    assert jnp.all(cv_after < 1e-8)


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZES))
def test_projection_with_only_box_constraint(seed: int, batch_size: int):
    """Test projection with only a box constraint and no nonlinear constraints."""
    key = jrnd.PRNGKey(seed)
    dim = 10

    mask_box = jnp.array(
        [True, True, True, True, False, False, False, False, False, False]
    )
    lb = jnp.array([[[-1.0], [-0.5], [-2.0], [-1.5]]])
    ub = jnp.array([[[1.0], [0.5], [2.0], [1.5]]])
    box = BoxConstraint(BoxConstraintSpecification(lb=lb, ub=ub, mask=mask_box))

    cartesian = CartesianConstraint(box_constraint=box, nl_constraints=None)

    key, subkey = jrnd.split(key)
    x = jrnd.uniform(subkey, shape=(batch_size, dim, 1), minval=-4, maxval=4)
    projection_instance = ProjectionInstance(x=x)

    cartesian_proj = cartesian.project(projection_instance)
    box_proj = box.project(projection_instance)

    # Cartesian projection should match direct box projection.
    assert jnp.allclose(cartesian_proj.x, box_proj.x)

    # Box dimensions should be clipped to bounds.
    assert jnp.all(cartesian_proj.x[:, 0:1, :] >= -1.0)
    assert jnp.all(cartesian_proj.x[:, 0:1, :] <= 1.0)
    assert jnp.all(cartesian_proj.x[:, 1:2, :] >= -0.5)
    assert jnp.all(cartesian_proj.x[:, 1:2, :] <= 0.5)
    assert jnp.all(cartesian_proj.x[:, 2:3, :] >= -2.0)
    assert jnp.all(cartesian_proj.x[:, 2:3, :] <= 2.0)
    assert jnp.all(cartesian_proj.x[:, 3:4, :] >= -1.5)
    assert jnp.all(cartesian_proj.x[:, 3:4, :] <= 1.5)

    # Unconstrained dimensions should remain unchanged.
    assert jnp.allclose(cartesian_proj.x[:, 4:, :], x[:, 4:, :])

    # CV for Cartesian with only box should equal box CV.
    assert jnp.allclose(cartesian.cv(projection_instance), box.cv(projection_instance))
