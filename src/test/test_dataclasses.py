"""This file contains unit tests for the dataclasses used in the Pinet layer."""

import re
from typing import cast

import jax.numpy as jnp
import pytest
from jaxtyping import TypeCheckError

from pinet import (
    BoxConstraintSpecification,
    EqualityConstraintsSpecification,
    EquilibrationParams,
    L2NormType,
    NonLinearSpecification,
    ProjectionInstance,
    SocConstraintSpecification,
    SOCType,
)
from pinet.constraints.non_linear_types import NonLinearConstraintType

# When ``PINET_RUNTIME_CHECK=1`` beartype catches malformed shapes/dtypes at
# ``__init__`` time with ``TypeCheckError``, pre-empting the library's own
# ``validate()`` which raises ``ValueError``. Tests that exercise either
# path accept both error types.
_ShapeOrValidationError = (ValueError, TypeError, TypeCheckError)


def test_eq_validate_requires_b_when_a_provided():
    spec = EqualityConstraintsSpecification(
        a_mat=jnp.ones((2, 5, 3)),  # (batch, n_constraints, dimension)
        b=None,
    )
    with pytest.raises(
        ValueError, match=re.escape("If a_mat is provided, b must also be provided.")
    ):
        spec.validate()


def test_eq_validate_passes_when_b_and_a_provided():
    spec = EqualityConstraintsSpecification(
        a_mat=jnp.ones((2, 5, 3)),
        b=jnp.ones((2, 5, 1)),
    )
    # should not raise
    spec.validate()


def test_eq_validate_passes_when_only_b_provided():
    # b without a_mat is allowed by current logic
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
    with pytest.raises(_ShapeOrValidationError):
        spec = BoxConstraintSpecification(lb=jnp.ones((2, 3)))  # wrong: 2D
        spec.validate()


def test_box_validate_ub_ndim_must_be_3():
    with pytest.raises(_ShapeOrValidationError):
        spec = BoxConstraintSpecification(ub=jnp.ones((2, 3)))  # wrong: 2D
        spec.validate()


def test_box_validate_lb_ub_same_nconstraints_required():
    with pytest.raises(_ShapeOrValidationError):
        lb = jnp.ones((4, 5, 1))
        ub = jnp.ones((4, 6, 1))  # n_constraints mismatch
        spec = BoxConstraintSpecification(lb=lb, ub=ub)
        spec.validate()


def test_box_validate_lb_ub_batch_mismatch_without_broadcast():
    with pytest.raises(_ShapeOrValidationError):
        lb = jnp.ones((4, 5, 1))
        ub = jnp.ones((3, 5, 1))  # batch mismatch and neither is 1
        spec = BoxConstraintSpecification(lb=lb, ub=ub)
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
    with pytest.raises(_ShapeOrValidationError):
        lb = jnp.ones((1, 3, 1))
        mask = jnp.array([1, 0, 1])  # int, not bool
        spec = BoxConstraintSpecification(lb=lb, mask=mask)
        spec.validate()


def test_box_validate_mask_must_be_1d():
    with pytest.raises(_ShapeOrValidationError):
        lb = jnp.ones((1, 3, 1))
        mask = jnp.array([[True, False, True]])  # 2D
        spec = BoxConstraintSpecification(lb=lb, mask=mask)
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
    with pytest.raises(_ShapeOrValidationError):
        x = jnp.ones((5, 3))  # 2D
        pi = ProjectionInstance(x=x)
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
    a_mat = jnp.ones((2, 3, 4))
    b = jnp.ones((2, 3, 1))
    apinv = jnp.ones((2, 4, 3))

    spec1 = spec0.update(a_mat=a_mat, b=b, a_mat_pinv=apinv)

    assert spec1 is not spec0, "update() should return a new equality specification."
    assert spec1.a_mat is a_mat, "Updated equality specification should store a_mat."
    assert spec1.b is b, "Updated equality specification should store b."
    assert spec1.a_mat_pinv is apinv, (
        "Updated equality specification should store a_mat_pinv."
    )
    # original remains unchanged
    assert spec0.a_mat is None and spec0.b is None and spec0.a_mat_pinv is None, (
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
        a_mat=jnp.ones((2, 1, 3)),
        b=jnp.ones((2, 1, 1)),
        a_mat_pinv=jnp.ones((2, 3, 1)),
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


def test_soc_validate_mask_u_must_be_boolean():
    with pytest.raises(_ShapeOrValidationError):
        mask_u = jnp.array([1, 0, 1, 0, 1])  # int, not bool
        mask_t = jnp.array([False, True, False], dtype=jnp.bool_)
        spec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t)
        spec.validate()


def test_soc_validate_mask_t_must_be_boolean():
    with pytest.raises(_ShapeOrValidationError):
        mask_u = jnp.array([True, False, True, False, True], dtype=jnp.bool_)
        mask_t = jnp.array([0, 1, 0])  # int, not bool
        spec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t)
        spec.validate()


def test_soc_validate_mask_u_must_be_1d():
    with pytest.raises(_ShapeOrValidationError):
        mask_u = jnp.array([[True, False], [True, False]], dtype=jnp.bool_)  # 2D
        mask_t = jnp.array([False, True, False], dtype=jnp.bool_)
        spec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t)
        spec.validate()


def test_soc_validate_mask_t_must_be_1d():
    with pytest.raises(_ShapeOrValidationError):
        mask_u = jnp.array([True, False, True, False, True], dtype=jnp.bool_)
        mask_t = jnp.array([[False, True], [False, False]], dtype=jnp.bool_)  # 2D
        spec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t)
        spec.validate()


def test_soc_validate_mask_u_mask_t_same_size():
    with pytest.raises(_ShapeOrValidationError):
        mask_u = jnp.array([True, False, True, False, True], dtype=jnp.bool_)
        mask_t = jnp.array([False, True, False], dtype=jnp.bool_)  # size = 3
        spec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t)
        spec.validate()


def test_soc_validate_mask_t_must_select_exactly_one():
    mask_u = jnp.array([True, False, True, False, True], dtype=jnp.bool_)
    mask_t = jnp.array(
        [False, True, True, False, False], dtype=jnp.bool_
    )  # sum = 2, not 1
    spec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t)
    with pytest.raises(ValueError, match=r"mask_t must select exactly one element."):
        spec.validate()


def test_soc_validate_mask_t_must_select_at_least_one():
    mask_u = jnp.array([True, False, True, False, True], dtype=jnp.bool_)
    mask_t = jnp.array([False, False, False, False, False], dtype=jnp.bool_)  # sum = 0
    spec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t)
    with pytest.raises(ValueError, match=r"mask_t must select exactly one element."):
        spec.validate()


def test_soc_validate_a_must_be_3d():
    with pytest.raises(_ShapeOrValidationError):
        mask_u = jnp.array([True, False, True], dtype=jnp.bool_)
        mask_t = jnp.array([False, True, False], dtype=jnp.bool_)
        a = jnp.ones((2, 3))  # 2D, not 3D
        spec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t, a=a)
        spec.validate()


def test_soc_validate_b_must_be_3d():
    with pytest.raises(_ShapeOrValidationError):
        mask_u = jnp.array([True, False, True], dtype=jnp.bool_)
        mask_t = jnp.array([False, True, False], dtype=jnp.bool_)
        b = jnp.ones((2, 1))  # 2D, not 3D
        spec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t, b=b)
        spec.validate()


def test_soc_validate_a_second_dim_must_match_mask_u_size():
    mask_u = jnp.array([True, False, True, False, True], dtype=jnp.bool_)  # 3 True values
    mask_t = jnp.array([False, True, False, False, False], dtype=jnp.bool_)
    a = jnp.ones((2, 5, 1))  # second dim = 5, but mask_u has only 3 True values
    spec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t, a=a)
    with pytest.raises(
        ValueError,
        match=(
            r"The second dimension of a must match the number of True values in mask_u."
        ),
    ):
        spec.validate()


def test_soc_validate_b_second_dim_must_be_1():
    with pytest.raises(_ShapeOrValidationError):
        mask_u = jnp.array([True, False, True], dtype=jnp.bool_)
        mask_t = jnp.array([False, True, False], dtype=jnp.bool_)
        b = jnp.ones((2, 3, 1))  # second dim = 3, not 1
        spec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t, b=b)
        spec.validate()


def test_soc_validate_passes_with_valid_inputs():
    mask_u = jnp.array([True, False, True, False, True], dtype=jnp.bool_)  # 3 True values
    mask_t = jnp.array([False, True, False, False, False], dtype=jnp.bool_)
    a = jnp.ones((2, 3, 1))  # matches number of True values in mask_u
    b = jnp.ones((2, 1, 1))  # second dim = 1
    spec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t, a=a, b=b)
    spec.validate()  # should not raise


def test_soc_validate_passes_with_minimal_inputs():
    mask_u = jnp.array([True, False], dtype=jnp.bool_)
    mask_t = jnp.array([False, True], dtype=jnp.bool_)
    spec = SocConstraintSpecification(mask_u=mask_u, mask_t=mask_t)
    spec.validate()  # should not raise


def test_nonlinear_validate_l2norm_with_rhs_not_supported():
    spec = NonLinearSpecification(
        nl_type=L2NormType,
        a_mat=jnp.ones((1, 2, 3)),
        f=jnp.ones((1, 1, 3)),
    )
    with pytest.raises(
        ValueError,
        match=r"L2NormType with RHS \(f\) is not supported in NonLinearSpecification.",
    ):
        spec.validate()


def test_nonlinear_validate_nl_type_must_be_constraint_type_instance():
    with pytest.raises(_ShapeOrValidationError):
        spec = NonLinearSpecification(
            nl_type=cast(NonLinearConstraintType, cast(object, "invalid_type")),
            a_mat=jnp.ones((1, 2, 3)),
        )
        spec.validate()


def test_validate_type_rejects_unsupported_nl_type_post_construction():
    """``_validate_type`` is the single owner of the supported-set
    invariant: the parser and ``to_primitive_spec`` rely on it and carry no
    guard of their own.

    The nl_type is corrupted *after* construction so beartype (hook on by
    default) does not pre-empt at the constructor; this exercises the
    reachable ``nl_type not in (SOCType, L2NormType)`` raise in-process.
    """
    spec = NonLinearSpecification(nl_type=SOCType, a_mat=jnp.ones((1, 2, 3)))
    object.__setattr__(spec, "nl_type", cast(object, "not_a_real_type"))
    with pytest.raises(ValueError, match=r"SOCType or L2NormType"):
        spec.validate()
    # to_primitive_spec routes through the same guard.
    with pytest.raises(ValueError, match=r"SOCType or L2NormType"):
        spec.to_primitive_spec()


def test_nonlinear_validate_inconsistent_batch_sizes_raises():
    with pytest.raises(_ShapeOrValidationError):
        spec = NonLinearSpecification(
            nl_type=SOCType,
            a_mat=jnp.ones((2, 2, 3)),
            a=jnp.ones((3, 2, 1)),
            f=jnp.ones((2, 1, 3)),
            b=jnp.ones((3, 1, 1)),
        )
        spec.validate()


def test_nonlinear_validate_batch_sizes_allow_broadcast_with_all_present():
    spec = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=jnp.ones((1, 2, 3)),
        a=jnp.ones((4, 2, 1)),
        f=jnp.ones((1, 1, 3)),
        b=jnp.ones((4, 1, 1)),
    )
    spec.validate()


def test_nonlinear_validate_A_or_f_batch_size_not_one():
    with pytest.raises(_ShapeOrValidationError):
        spec = NonLinearSpecification(
            nl_type=SOCType,
            a_mat=jnp.ones((2, 2, 3)),
            a=jnp.ones((1, 2, 1)),
            f=jnp.ones((1, 1, 3)),
            b=jnp.ones((2, 1, 1)),
        )
        spec.validate()

    with pytest.raises(_ShapeOrValidationError):
        spec = NonLinearSpecification(
            nl_type=SOCType,
            a_mat=jnp.ones((1, 2, 3)),
            a=jnp.ones((1, 2, 1)),
            f=jnp.ones((2, 1, 3)),
            b=jnp.ones((2, 1, 1)),
        )
        spec.validate()


def test_nonlinear_validate_A_and_a_constraint_dimension():
    with pytest.raises(_ShapeOrValidationError):
        spec = NonLinearSpecification(
            nl_type=SOCType,
            a_mat=jnp.ones((1, 3, 3)),
            a=jnp.ones((1, 2, 1)),
            f=jnp.ones((1, 1, 3)),
            b=jnp.ones((2, 1, 1)),
        )
        spec.validate()


def test_nonlinear_validate_A_and_f_variable_dimension():
    with pytest.raises(_ShapeOrValidationError):
        spec = NonLinearSpecification(
            nl_type=SOCType,
            a_mat=jnp.ones((1, 3, 3)),
            a=jnp.ones((1, 3, 1)),
            f=jnp.ones((1, 1, 2)),
            b=jnp.ones((2, 1, 1)),
        )
        spec.validate()


def test_nonlinear_validate_f_and_b_constraint_dimension():
    with pytest.raises(_ShapeOrValidationError):
        spec = NonLinearSpecification(
            nl_type=SOCType,
            a_mat=jnp.ones((1, 3, 3)),
            a=jnp.ones((1, 3, 1)),
            f=jnp.ones((1, 2, 3)),
            b=jnp.ones((2, 1, 1)),
        )
        spec.validate()


def test_nonlinear_validate_b_is_not_a_scalar():
    with pytest.raises(_ShapeOrValidationError):
        spec = NonLinearSpecification(
            nl_type=SOCType,
            a_mat=jnp.ones((1, 3, 3)),
            a=jnp.ones((1, 3, 1)),
            f=jnp.ones((1, 2, 3)),
            b=jnp.ones((2, 2, 1)),
        )
        spec.validate()


def test_nonlinear_to_primitive_spec_with_invalid_type():
    # to_primitive_spec validates the type up front: an unsupported nl_type
    # fails via _validate_type (ValueError) or, with PINET_RUNTIME_CHECK=1,
    # is rejected by beartype at construction (TypeCheckError).
    with pytest.raises((ValueError, TypeCheckError)):
        spec = NonLinearSpecification(
            nl_type=cast(NonLinearConstraintType, cast(object, "invalid_type")),
            a_mat=jnp.ones((1, 2, 3)),
        )
        spec.to_primitive_spec()
