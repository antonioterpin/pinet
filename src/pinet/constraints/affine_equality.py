"""Equality constraint module."""

import equinox as eqx
import jax.numpy as jnp

from pinet._typing import BatchedEqMatrix, BatchedEqPinv, BatchedRHS, BatchedScalar
from pinet.dataclasses import ProjectionInstance

from .base import Constraint

_EXPECTED_NDIM = 3
_LAST_AXIS_SIZE_B = 1


class EqualityConstraint(Constraint):
    """Equality constraint set.

    The (affine) equality constraint set is defined as:
    a @ x == b
    where the matrix a and the vector b are the parameters.

    Attributes:
        a: Left-hand side matrix.
        b: Right-hand side vector.
        a_pinv: Pseudoinverse of ``a`` (cached when ``var_a`` is False).
        method: Solver strategy; ``"pinv"`` or ``None``.
        var_b: Whether ``b`` is expected to vary per instance.
        var_a: Whether ``a`` is expected to vary per instance.
    """

    a: BatchedEqMatrix
    b: BatchedRHS
    a_pinv: BatchedEqPinv | None = None
    method: str | None = eqx.field(static=True, default="pinv")
    var_b: bool | None = eqx.field(static=True, default=False)
    var_a: bool | None = eqx.field(static=True, default=False)

    def __init__(
        self,
        a: BatchedEqMatrix,
        b: BatchedRHS,
        method: str | None = "pinv",
        var_b: bool | None = False,
        var_a: bool | None = False,
    ) -> None:
        """Initialize the equality constraint.

        Args:
            a: Left hand side matrix.
            b: Right hand side vector.
            method: String that specifies the method used to solve
                linear systems. Valid methods are "pinv", and None.
            var_b: Boolean that indicates whether the b vector
                changes or is constant.
            var_a: Boolean that indicates whether the a matrix
                changes or is constant.
        """
        # The equality constraint always needs its left-hand side matrix.
        assert a is not None, "Matrix a must be provided."
        # The equality constraint always needs its right-hand side vector.
        assert b is not None, "Vector b must be provided."

        # The equality matrix is batched as (batch_size, n_constraints, dimension).
        assert a.ndim == _EXPECTED_NDIM, (
            "a is a matrix with shape (batch_size, n_constraints, dimension)."
        )
        # The right-hand side is batched with the same rank as a.
        assert b.ndim == _EXPECTED_NDIM, (
            "b is a matrix with shape (batch_size, n_constraints, 1)."
        )
        # The last axis of b stores a single scalar per constraint.
        assert b.shape[2] == _LAST_AXIS_SIZE_B, (
            "b must have shape (batch_size, n_constraints, 1)."
        )
        # Batch sizes must be the same, or one of them must be 1.
        assert a.shape[0] == b.shape[0] or a.shape[0] == 1 or b.shape[0] == 1, (
            f"Batch sizes are inconsistent: a{a.shape}, b{b.shape}"
        )
        # Each equality row in a needs one matching entry in b.
        assert a.shape[1] == b.shape[1], "Number of rows in a must equal size of b."

        valid_methods = ["pinv", None]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method {method}. Valid methods are: {valid_methods}"
            )

        a_pinv: BatchedEqPinv | None = None
        if method == "pinv" and not var_a:
            a_pinv = jnp.linalg.pinv(a)

        self.a = a
        self.b = b
        self.a_pinv = a_pinv
        self.method = method
        self.var_b = var_b
        self.var_a = var_a

    def get_params(
        self, inp: ProjectionInstance
    ) -> tuple[BatchedRHS, BatchedEqMatrix | None, BatchedEqPinv | None]:
        """Get matrix, b, matrix_pinv depending on varying constraints.

        Args:
            inp: ProjectionInstance to get parameters from.

        Returns:
            Tuple ``(b, matrix, matrix_pinv)`` with the right-hand side vector,
            the left-hand side matrix and its pseudo-inverse, respectively.
        """
        b = inp.eq.b if inp.eq and inp.eq.b is not None else self.b
        a = inp.eq.a if inp.eq and self.var_a else self.a
        a_pinv = inp.eq.a_pinv if inp.eq and self.var_a else self.a_pinv
        return b, a, a_pinv

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project onto equality constraints.

        Args:
            yraw: ProjectionInstance to projection.
                The .x attribute is the point to project.

        Returns:
            The projected point for each point in the batch.

        Raises:
            NotImplementedError: If ``method`` is ``None``.
        """
        if self.method is None:
            raise NotImplementedError("No projection method set.")
        return self.project_pinv(yraw)

    def project_pinv(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project onto equality constraints using pseudo-inverse.

        Args:
            yraw: ProjectionInstance to projection.
                The .x attribute is the point to project.

        Returns:
            The projected point for each point in the batch.
        """
        b, a, a_pinv = self.get_params(yraw)
        # a must be available to apply the projection.
        assert a is not None, (
            "a must be provided in EqualityConstraintsSpecification "
            "when var_a=True and a_pinv is not supplied."
        )
        if a_pinv is None:
            a_pinv = jnp.linalg.pinv(a)

        return yraw.update(x=yraw.x - a_pinv @ (a @ yraw.x - b))

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self.a.shape[-1]

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.a.shape[1]

    def cv(self, yraw: ProjectionInstance) -> BatchedScalar:
        """Compute the constraint violation.

        Args:
            yraw: ProjectionInstance to evaluate.

        Returns:
            The constraint violation for each point in the batch.
        """
        b, a, _ = self.get_params(yraw)
        # a must be available to compute the violation.
        assert a is not None

        return jnp.linalg.norm(a @ yraw.x - b, ord=jnp.inf, axis=1, keepdims=True)
