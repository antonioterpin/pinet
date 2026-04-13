"""Abstract class for non-linear constraints."""

from typing import Optional

import jax.numpy as jnp

from pinet.constraints.base import Constraint
from pinet.constraints.non_linear_types import NonLinearConstraintType
from pinet.dataclasses import NonLinearSpecification


class NonLinearConstraint(Constraint):
    """Abstract class for non-linear constraints.

    This class describes constraints of the form:
    g(A @ y + a) <= f @ y + b,
    where g is a convex function.
    """

    def __init__(
        self,
        spec: NonLinearSpecification,
    ) -> None:
        """Initialize the non-linear constraint.

        Args:
            spec (NonLinearSpecification): Specification of the non-linear constraint.
        """
        # Check for dimension
        spec.validate()
        self._dim = spec.A.shape[-1]

        self.spec = spec
        self._A = spec.A
        self._a = spec.a
        self._f = spec.f
        self._b = spec.b
        self._nl_type = spec.nl_type

    @property
    def A(self) -> Optional[jnp.ndarray]:
        """Matrix for linear transformation before non-linear function g.

        Returns:
            Optional[jnp.ndarray]: Shape (batch_size, constraint_dim, variable_dim)
        """
        return self._A

    @property
    def a(self) -> Optional[jnp.ndarray]:
        """Offset vector before non-linear function g.

        Returns:
            Optional[jnp.ndarray]: Shape (batch_size, constraint_dim, 1)
        """
        return self._a

    @property
    def f(self) -> Optional[jnp.ndarray]:
        """Linear coefficients on the right-hand side.

        Returns:
            Optional[jnp.ndarray]: Shape (batch_size, 1, variable_dim)
        """
        return self._f

    @property
    def b(self) -> Optional[jnp.ndarray]:
        """Constant offset on the right-hand side.

        Returns:
            Optional[jnp.ndarray]: Shape (batch_size, 1, 1)
        """
        return self._b

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return 1

    @property
    def nl_type(self) -> NonLinearConstraintType:
        """Return the type of non-linear constraint.

        Returns:
            NonLinearConstraintType: The type of non-linear constraint.
        """
        return self._nl_type
