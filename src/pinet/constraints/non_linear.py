"""Abstract class for non-linear constraints."""

import equinox as eqx

from pinet._typing import BatchedRHS, BatchedScalar, NLLinearRHS, NLMatrix
from pinet.constraints.base import Constraint
from pinet.constraints.non_linear_types import NonLinearConstraintType
from pinet.dataclasses import NonLinearSpecification, ProjectionInstance


class NonLinearConstraint(Constraint):
    """Abstract class for non-linear constraints.

    This class describes constraints of the form:
    g(A @ y + a) <= f @ y + b,
    where ``A`` is ``a_mat``, ``a`` is ``a``, ``f`` is ``f``, ``b`` is ``b``,
    and ``g`` is a convex function. Concrete non-linear constraints (e.g.
    ``SocConstraint``) provide ``project`` and ``cv``; calling these on the base
    class raises ``NotImplementedError``.

    Attributes:
        spec: Non-linear constraint specification.
    """

    spec: NonLinearSpecification
    _a_mat: NLMatrix
    _a: BatchedRHS | None
    _f: NLLinearRHS | None
    _b: BatchedScalar | None
    _nl_type: NonLinearConstraintType = eqx.field(static=True)
    _dim: int = eqx.field(static=True)

    def __init__(
        self,
        spec: NonLinearSpecification,
    ) -> None:
        """Initialize the non-linear constraint.

        Args:
            spec: Specification of the non-linear constraint.
        """
        # Check for dimension
        spec.validate()
        self._dim = spec.a_mat.shape[-1]

        self.spec = spec
        self._a_mat = spec.a_mat
        self._a = spec.a
        self._f = spec.f
        self._b = spec.b
        self._nl_type = spec.nl_type

    @property
    def a_mat(self) -> NLMatrix:
        """Matrix for linear transformation before non-linear function g."""
        return self._a_mat

    @property
    def a(self) -> BatchedRHS | None:
        """Offset vector before non-linear function g."""
        return self._a

    @property
    def f(self) -> NLLinearRHS | None:
        """Linear coefficients on the right-hand side."""
        return self._f

    @property
    def b(self) -> BatchedScalar | None:
        """Constant offset on the right-hand side."""
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
        """Return the type of non-linear constraint."""
        return self._nl_type

    def project(self, y_raw: ProjectionInstance) -> ProjectionInstance:
        """Project the input to the feasible region.

        Args:
            y_raw: ProjectionInstance to project.

        Returns:
            The projected input.

        Raises:
            NotImplementedError: Always; the base class provides only the shared
                parameter carrier. Use a concrete subclass (``SocConstraint``).
        """
        raise NotImplementedError(
            "NonLinearConstraint is a parameter carrier; use a concrete subclass."
        )

    def cv(self, y_raw: ProjectionInstance) -> BatchedScalar:
        """Compute the constraint violation.

        Args:
            y_raw: ProjectionInstance to evaluate.

        Returns:
            The constraint violation for each point in the batch.

        Raises:
            NotImplementedError: Always; the base class provides only the shared
                parameter carrier. Use a concrete subclass (``SocConstraint``).
        """
        raise NotImplementedError(
            "NonLinearConstraint is a parameter carrier; use a concrete subclass."
        )
