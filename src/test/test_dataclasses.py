"""This file contains unit tests for the dataclasses used in the Pinet layer."""

import re

import jax.numpy as jnp
import pytest

from pinet import (
    BoxConstraintSpecification,
    EqualityConstraintsSpecification,
    EquilibrationParams,
    ProjectionInstance,
)


def test_eq_validate_requires_b_when_a_provided():
    spec = EqualityConstraintsSpecification(
        a_dyn=jnp.ones((2, 5, 3)),  # (batch, n_constraints, dimension)
        b=None,
    )
    with pytest.raises(
        ValueError, match=re.escape("If a_dyn is provided, b must also be provided.")
    ):
        spec.validate()


def test_eq_validate_passes_when_b_and_a_provided():
    spec = EqualityConstraintsSpecification(
        a_dyn=jnp.ones((2, 5, 3)),
        b=jnp.ones((2, 5, 1)),
    )
    # should not raise
    spec.validate()


def test_eq_validate_passes_when_only_b_provided():
    # b without a_dyn is allowed by current logic
    spec = EqualityConstraintsSpecification(b=jnp.ones((1, 3, 1)))
    spec.validate()


def test_eq_validate_passes_when_both_none():
    spec = EqualityConstraintsSpecification()
    spec.validate()


def test_box_validate_requires_at_least_one_bound():
    spec = BoxConstraintSpecification(lb=None, ub=None)
    with pytest.raises(
        ValueError,
        match=re.escape("At least one of lower or upper bounds must be provided."),
    ):
        spec.validate()


def test_box_validate_lb_ndim_must_be_3():
    spec = BoxConstraintSpecification(lb=jnp.ones((2, 3)))  # wrong: 2D
    with pytest.raises(ValueError, match="Lower bound must have shape"):
        spec.validate()


def test_box_validate_ub_ndim_must_be_3():
    spec = BoxConstraintSpecification(ub=jnp.ones((2, 3)))  # wrong: 2D
    with pytest.raises(ValueError, match="Upper bound must have shape"):
        spec.validate()


def test_box_validate_lb_ub_same_nconstraints_required():
    lb = jnp.ones((4, 5, 1))
    ub = jnp.ones((4, 6, 1))  # n_constraints mismatch
    spec = BoxConstraintSpecification(lb=lb, ub=ub)
    with pytest.raises(
        ValueError, match="Lower and upper bounds must have the same shape"
    ):
        spec.validate()


def test_box_validate_lb_ub_batch_mismatch_without_broadcast():
    lb = jnp.ones((4, 5, 1))
    ub = jnp.ones((3, 5, 1))  # batch mismatch and neither is 1
    spec = BoxConstraintSpecification(lb=lb, ub=ub)
    with pytest.raises(
        ValueError, match="Batch size of lower and upper bounds must be the same"
    ):
        spec.validate()


def test_box_validate_lb_must_be_le_ub():
    lb = jnp.array([1.0, 2.0]).reshape(1, 2, 1)
    ub = jnp.array([0.5, 3.0]).reshape(1, 2, 1)  # first entry violates lb <= ub
    spec = BoxConstraintSpecification(lb=lb, ub=ub)
    with pytest.raises(
        ValueError,
        match=re.escape("Lower bound must be less than or equal to the upper bound."),
    ):
        spec.validate()


def test_box_validate_mask_dtype_bool_required():
    lb = jnp.ones((1, 3, 1))
    mask = jnp.array([1, 0, 1])  # int, not bool
    spec = BoxConstraintSpecification(lb=lb, mask=mask)
    with pytest.raises(TypeError, match=re.escape("Mask must be a boolean array.")):
        spec.validate()


def test_box_validate_mask_must_be_1d():
    lb = jnp.ones((1, 3, 1))
    mask = jnp.array([[True, False, True]])  # 2D
    spec = BoxConstraintSpecification(lb=lb, mask=mask)
    with pytest.raises(ValueError, match=re.escape("Mask must be a 1D array.")):
        spec.validate()


def test_box_validate_mask_active_count_matches_bounds():
    lb = jnp.ones((1, 3, 1))  # n_constraints = 3
    mask = jnp.array([True, False])  # sum = 1 != 3
    spec = BoxConstraintSpecification(lb=lb, mask=mask)
    with pytest.raises(
        ValueError,
        match=re.escape("Number of active entries in the mask must match the bounds."),
    ):
        spec.validate()


def test_box_validate_valid_lb_only():
    lb = jnp.ones((2, 4, 1))
    spec = BoxConstraintSpecification(lb=lb)
    spec.validate()  # should not raise


def test_box_validate_valid_ub_only():
    ub = jnp.zeros((3, 2, 1))
    spec = BoxConstraintSpecification(ub=ub)
    spec.validate()  # should not raise


def test_box_validate_valid_both_with_broadcast_and_mask():
    lb = jnp.zeros((1, 4, 1))  # batch=1 broadcastable
    ub = jnp.ones((3, 4, 1))  # batch=3
    mask = jnp.array([True, True, True, True])  # sum = 4 == n_constraints
    spec = BoxConstraintSpecification(lb=lb, ub=ub, mask=mask)
    spec.validate()  # should not raise


def test_projection_validate_x_ndim_must_be_3():
    x = jnp.ones((5, 3))  # 2D
    pi = ProjectionInstance(x=x)
    with pytest.raises(ValueError, match="x must have shape"):
        pi.validate()


def test_projection_validate_passes_when_x_is_3d():
    x = jnp.ones((2, 3, 1))
    pi = ProjectionInstance(x=x)
    pi.validate()  # should not raise


def test_equilibration_validate_accepts_defaults():
    # Default values should pass
    EquilibrationParams().validate()


def test_equilibration_validate_max_iter_non_negative():
    with pytest.raises(ValueError, match=re.escape("max_iter must be non-negative.")):
        EquilibrationParams(max_iter=-1).validate()


@pytest.mark.parametrize("tol", [0.0, -1e-6])
def test_equilibration_validate_tol_positive(tol):
    with pytest.raises(ValueError, match=re.escape("tol must be positive.")):
        EquilibrationParams(tol=tol).validate()


@pytest.mark.parametrize("ord_val", [1, 2, float("inf")])
def test_equilibration_validate_ord_allowed(ord_val):
    # Allowed ord values should pass
    EquilibrationParams(ord=ord_val).validate()


def test_equilibration_validate_ord_invalid_raises():
    with pytest.raises(ValueError, match=re.escape("ord must be 1, 2, or infinity.")):
        EquilibrationParams(ord=3).validate()


@pytest.mark.parametrize("mode", ["Gauss", "Jacobi"])
def test_equilibration_validate_update_mode_allowed(mode):
    # Both modes should pass
    EquilibrationParams(update_mode=mode).validate()


def test_equilibration_validate_update_mode_invalid_raises():
    with pytest.raises(
        ValueError, match=re.escape('update_mode must be either "Gauss" or "Jacobi".')
    ):
        EquilibrationParams(update_mode="Foo").validate()


def test_eq_update_returns_new_and_sets_fields():
    spec0 = EqualityConstraintsSpecification()
    a_dyn = jnp.ones((2, 3, 4))
    b = jnp.ones((2, 3, 1))
    apinv = jnp.ones((2, 4, 3))

    spec1 = spec0.update(a_dyn=a_dyn, b=b, a_dyn_pinv=apinv)

    assert spec1 is not spec0, "update() should return a new equality specification."
    assert spec1.a_dyn is a_dyn, "Updated equality specification should store a_dyn."
    assert spec1.b is b, "Updated equality specification should store b."
    assert spec1.a_dyn_pinv is apinv, (
        "Updated equality specification should store a_dyn_pinv."
    )
    # original remains unchanged
    assert spec0.a_dyn is None and spec0.b is None and spec0.a_dyn_pinv is None, (
        "update() should not mutate the original equality specification."
    )


def test_eq_update_unknown_kw_raises_typeerror():
    spec0 = EqualityConstraintsSpecification()
    with pytest.raises(TypeError):
        spec0.update(foo="bar")


def test_box_update_returns_new_and_sets_fields():
    spec0 = BoxConstraintSpecification()
    lb = jnp.zeros((1, 2, 1))
    ub = jnp.ones((3, 2, 1))
    mask = jnp.array([True, False])

    spec1 = spec0.update(lb=lb, ub=ub, mask=mask)

    assert spec1 is not spec0, "update() should return a new box specification."
    assert spec1.lb is lb, "Updated box specification should store lb."
    assert spec1.ub is ub, "Updated box specification should store ub."
    assert spec1.mask is mask, "Updated box specification should store mask."
    # original remains unchanged
    assert spec0.lb is None and spec0.ub is None and spec0.mask is None, (
        "update() should not mutate the original box specification."
    )


def test_box_update_unknown_kw_raises_typeerror():
    spec0 = BoxConstraintSpecification()
    with pytest.raises(TypeError):
        spec0.update(invalid_field=123)


def test_projection_update_sets_eq_and_box_and_returns_new():
    x0 = jnp.ones((2, 3, 1))
    pi0 = ProjectionInstance(x=x0)

    eq = EqualityConstraintsSpecification(
        a_dyn=jnp.ones((2, 1, 3)),
        b=jnp.ones((2, 1, 1)),
        a_dyn_pinv=jnp.ones((2, 3, 1)),
    )
    box = BoxConstraintSpecification(
        lb=jnp.zeros((2, 3, 1)),
        ub=jnp.ones((2, 3, 1)),
        mask=jnp.array([True, True, True]),
    )

    pi1 = pi0.update(eq=eq, box=box)

    assert pi1 is not pi0, "update() should return a new projection instance."
    assert pi1.eq is eq, "Updated projection instance should store the equality spec."
    assert pi1.box is box, "Updated projection instance should store the box spec."
    # original remains unchanged
    assert pi0.eq is None and pi0.box is None, (
        "update() should not mutate the original projection instance."
    )


def test_projection_update_x_and_returns_new():
    x0 = jnp.ones((2, 3, 1))
    x1 = jnp.zeros((2, 3, 1))
    pi0 = ProjectionInstance(x=x0)
    pi1 = pi0.update(x=x1)

    assert pi1 is not pi0, "update() should return a new projection instance."
    assert (pi1.x == x1).all(), "Updated projection instance should store the new x."
    assert (pi0.x == x0).all(), "update() should not mutate the original x."


def test_projection_update_unknown_kw_raises_typeerror():
    pi0 = ProjectionInstance(x=jnp.ones((1, 1, 1)))
    with pytest.raises(TypeError):
        pi0.update(does_not_exist=True)


def test_equilibration_update_changes_fields_and_returns_new():
    expected_max_iter = 10
    default_ord = 2
    ep0 = EquilibrationParams()
    ep1 = ep0.update(
        max_iter=expected_max_iter,
        tol=1e-4,
        ord=1,
        col_scaling=True,
        update_mode="Jacobi",
        safeguard=True,
    )

    assert ep1 is not ep0, "update() should return a new equilibration parameter set."
    assert ep1.max_iter == expected_max_iter, (
        "Updated equilibration parameters should store max_iter."
    )
    assert ep1.tol == pytest.approx(1e-4), (
        "Updated equilibration parameters should store tol."
    )
    assert ep1.ord == 1, "Updated equilibration parameters should store ord."
    assert ep1.col_scaling is True, (
        "Updated equilibration parameters should store col_scaling."
    )
    assert ep1.update_mode == "Jacobi", (
        "Updated equilibration parameters should store update_mode."
    )
    assert ep1.safeguard is True, (
        "Updated equilibration parameters should store safeguard."
    )

    # original remains defaults
    assert ep0.max_iter == 0, "update() should not mutate the original max_iter."
    assert ep0.tol == pytest.approx(1e-3), "update() should not mutate the original tol."
    assert ep0.ord == default_ord, "update() should not mutate the original ord."
    assert ep0.col_scaling is False, (
        "update() should not mutate the original col_scaling."
    )
    assert ep0.update_mode == "Gauss", (
        "update() should not mutate the original update_mode."
    )
    assert ep0.safeguard is False, "update() should not mutate the original safeguard."


def test_equilibration_update_unknown_kw_raises_typeerror():
    ep0 = EquilibrationParams()
    with pytest.raises(TypeError):
        ep0.update(foo="bar")
