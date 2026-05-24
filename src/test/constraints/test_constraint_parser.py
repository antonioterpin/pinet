"""Unit tests for ``ConstraintParser`` behavior itself.

These exercise the parser's structural contracts — identity passthrough when
no lifting is needed and construction-time validation — independent of the
end-to-end projection-equivalence suites in ``test_parser_and_affine_inequality``
and ``test_parser_and_nonlinear``.
"""

import jax
import jax.numpy as jnp
import pytest

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    BoxConstraintSpecification,
    ConstraintParser,
    EqualityConstraint,
    NonLinearConstraint,
    NonLinearSpecification,
    SOCType,
)

jax.config.update("jax_enable_x64", True)


def test_constraint_parser_no_ineq_no_box_returns_eq_as_is():
    dim, n_eq = 3, 2
    a_mat = jnp.arange(n_eq * dim, dtype=jnp.float64).reshape(1, n_eq, dim)
    b = jnp.zeros((1, n_eq, 1))
    eq = EqualityConstraint(a_mat=a_mat, b=b, method="pinv")

    parser = ConstraintParser(eq_constraint=eq, ineq_constraint=None, box_constraint=None)
    eq_out, box_out, _ = parser.parse(method="pinv")

    # Still the same exact object (no lifting performed)
    assert eq_out is eq, (
        "Parser should return the original equality constraint when no lifting is needed."
    )
    assert box_out is None, (
        "Parser should not synthesize a box constraint when none is provided."
    )
    assert eq_out is not None, (
        "Parser should preserve the equality constraint when no lifting is needed."
    )
    assert eq_out.a_mat is a_mat, (
        "Parser should preserve the original equality matrix when no lifting is needed."
    )
    assert eq_out.b is b, (
        "Parser should preserve the original equality RHS when no lifting is needed."
    )


def test_constraint_parser_no_ineq_with_box_returns_inputs():
    dim, n_eq = 4, 1
    a_mat = jnp.ones((1, n_eq, dim))
    b = jnp.zeros((1, n_eq, 1))
    eq = EqualityConstraint(a_mat=a_mat, b=b, method="pinv")

    mask = jnp.array([True, False, True, False])
    n_box = int(mask.sum())
    lb = jnp.array([[[-1.0], [0.0]]]).reshape(1, n_box, 1)
    ub = jnp.array([[[1.0], [2.0]]]).reshape(1, n_box, 1)
    box = BoxConstraint(BoxConstraintSpecification(lb=lb, ub=ub, mask=mask))

    parser = ConstraintParser(eq_constraint=eq, ineq_constraint=None, box_constraint=box)
    eq_out, box_out, _ = parser.parse(method="pinv")

    # Still the same exact objects (no lifting performed)
    assert eq_out is eq, (
        "Parser should return the original equality constraint when only a box "
        "constraint is provided."
    )
    assert box_out is box, (
        "Parser should return the original box constraint when no lifting is needed."
    )
    assert isinstance(box_out, BoxConstraint), (
        "Parser should preserve the box constraint when no lifting is needed."
    )
    assert box_out.mask is not None, (
        "Preserved box constraint should still expose its mask."
    )
    assert box_out.lb is not None, (
        "Preserved box constraint should still expose its lower bounds."
    )
    assert box_out.ub is not None, (
        "Preserved box constraint should still expose its upper bounds."
    )

    # Sanity: mask/bounds unchanged
    assert jnp.array_equal(box_out.mask, mask), (
        "Parser should preserve the original box mask when no lifting is needed."
    )
    assert jnp.array_equal(box_out.lb, lb), (
        "Parser should preserve the original box lower bounds when no lifting is needed."
    )
    assert jnp.array_equal(box_out.ub, ub), (
        "Parser should preserve the original box upper bounds when no lifting is needed."
    )


def test_parse_non_linear_with_ineq_batch_size_not_one_raises():
    """Test parser raises when inequality C batch size is not 1 in nonlinear mode."""
    eq_constraint = EqualityConstraint(
        a_mat=jnp.array([[[1.0, 0.0, 0.0]]]),
        b=jnp.array([[[0.0]]]),
        var_b=False,
    )

    # Nonlinear parsing requires C batch size == 1; use batch size 2 to trigger.
    c_mat = jnp.array(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
        ]
    )
    lb = jnp.zeros((2, 1, 1))
    ub = jnp.ones((2, 1, 1))
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)

    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=jnp.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        a=jnp.zeros((1, 2, 1)),
        f=jnp.array([[[0.0, 0.0, 1.0]]]),
        b=jnp.ones((1, 1, 1)),
    )
    nl_constraint = NonLinearConstraint(spec=nl_spec)

    with pytest.raises(
        AssertionError,
        match=r"Batch size of inequality constraint must be 1 or None",
    ):
        ConstraintParser(
            eq_constraint=eq_constraint,
            ineq_constraint=ineq_constraint,
            box_constraint=None,
            nl_constraints=[nl_constraint],
        )
