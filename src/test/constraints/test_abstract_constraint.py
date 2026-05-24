"""Tests for the abstract ``Constraint`` base class and its direct subclasses.

These exercise the fallback / abstract branches that concrete constraint
subclasses (``EqualityConstraint``, ``BoxConstraint``, ``SocConstraint``,
``CartesianConstraint``) never hit, keeping the library coverage above the
95% threshold.
"""

import jax.numpy as jnp
import pytest

from pinet import (
    EqualityConstraint,
    EqualityConstraintsSpecification,
    NonLinearSpecification,
    ProjectionInstance,
)
from pinet.constraints.base import Constraint
from pinet.constraints.non_linear import NonLinearConstraint
from pinet.constraints.non_linear_types import SOCType


def test_constraint_base_project_raises() -> None:
    """``Constraint.project`` must fail with a named NotImplementedError."""
    c = Constraint()
    y_raw = ProjectionInstance(x=jnp.zeros((1, 1, 1)))
    with pytest.raises(NotImplementedError, match=r"Constraint\.project"):
        c.project(y_raw)


def test_constraint_base_cv_raises() -> None:
    """``Constraint.cv`` must fail with a named NotImplementedError."""
    c = Constraint()
    y_raw = ProjectionInstance(x=jnp.zeros((1, 1, 1)))
    with pytest.raises(NotImplementedError, match=r"Constraint\.cv"):
        c.cv(y_raw)


def test_constraint_base_dim_raises() -> None:
    """``Constraint.dim`` must fail with a named NotImplementedError."""
    c = Constraint()
    with pytest.raises(NotImplementedError, match=r"Constraint\.dim"):
        _ = c.dim


def test_constraint_base_n_constraints_raises() -> None:
    """``Constraint.n_constraints`` must fail with a named NotImplementedError."""
    c = Constraint()
    with pytest.raises(NotImplementedError, match=r"Constraint\.n_constraints"):
        _ = c.n_constraints


@pytest.fixture
def nl_parameter_carrier() -> NonLinearConstraint:
    """Provide a minimal valid ``NonLinearConstraint`` parameter carrier."""
    spec = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=jnp.zeros((1, 1, 2)),
        a=jnp.zeros((1, 1, 1)),
        f=None,
        b=jnp.zeros((1, 1, 1)),
    )
    return NonLinearConstraint(spec)


def test_nonlinear_constraint_n_constraints_is_one(
    nl_parameter_carrier: NonLinearConstraint,
) -> None:
    """Direct instantiation of ``NonLinearConstraint`` reports a single constraint."""
    assert nl_parameter_carrier.n_constraints == 1


def test_nonlinear_constraint_project_raises(
    nl_parameter_carrier: NonLinearConstraint,
) -> None:
    """``NonLinearConstraint.project`` signals the subclass contract."""
    y_raw = ProjectionInstance(x=jnp.zeros((1, 2, 1)))
    with pytest.raises(NotImplementedError, match=r"parameter carrier"):
        nl_parameter_carrier.project(y_raw)


def test_nonlinear_constraint_cv_raises(
    nl_parameter_carrier: NonLinearConstraint,
) -> None:
    """``NonLinearConstraint.cv`` signals the subclass contract."""
    y_raw = ProjectionInstance(x=jnp.zeros((1, 2, 1)))
    with pytest.raises(NotImplementedError, match=r"parameter carrier"):
        nl_parameter_carrier.cv(y_raw)


def test_equality_project_pinv_recomputes_when_pinv_missing() -> None:
    """``project_pinv`` falls back to ``jnp.linalg.pinv`` when no pinv is cached.

    Exercises the ``a_mat_pinv is None`` branch inside ``project_pinv``: the
    constraint is configured with ``var_a_mat=True`` and the per-instance
    spec supplies ``a_mat`` but no ``a_mat_pinv``.
    """
    a_mat = jnp.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    b = jnp.array([[[0.3], [-0.4]]])
    eq = EqualityConstraint(a_mat=a_mat, b=b, method="pinv", var_a_mat=True)

    x = jnp.array([[[1.0], [2.0], [3.0]]])
    y_raw = ProjectionInstance(
        x=x,
        eq=EqualityConstraintsSpecification(a_mat=a_mat, b=b, a_mat_pinv=None),
    )
    out = eq.project_pinv(y_raw)
    # The first two coordinates must match b exactly after projection.
    assert jnp.allclose(out.x[0, 0, 0], 0.3)
    assert jnp.allclose(out.x[0, 1, 0], -0.4)
    # The unconstrained coordinate stays untouched.
    assert jnp.allclose(out.x[0, 2, 0], 3.0)
