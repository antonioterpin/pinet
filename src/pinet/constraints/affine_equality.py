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
    a_mat @ x == b
    where the matrix a_mat and the vector b are the parameters.

    Attributes:
        a_mat: Left-hand side matrix.
        b: Right-hand side vector.
        a_mat_pinv: Pseudoinverse of ``a_mat`` (cached when ``var_a_mat`` is False).
        method: Solver strategy; ``"pinv"`` or ``None``.
        var_b: Whether ``b`` is expected to vary per instance.
        var_a_mat: Whether ``a_mat`` is expected to vary per instance.
    """

    a_mat: BatchedEqMatrix
    b: BatchedRHS
    a_mat_pinv: BatchedEqPinv | None = None
    method: str | None = eqx.field(static=True, default="pinv")
    var_b: bool | None = eqx.field(static=True, default=False)
    var_a_mat: bool | None = eqx.field(static=True, default=False)

    def __init__(
        self,
        a_mat: BatchedEqMatrix,
        b: BatchedRHS,
        method: str | None = "pinv",
        var_b: bool | None = False,
        var_a_mat: bool | None = False,
    ) -> None:
        """Initialize the equality constraint.

        Args:
            a_mat: Left hand side matrix.
            b: Right hand side vector.
            method: String that specifies the method used to solve
                linear systems. Valid methods are "pinv", and None.
            var_b: Boolean that indicates whether the b vector
                changes or is constant.
            var_a_mat: Boolean that indicates whether the a_mat matrix
                changes or is constant.
        """
        assert a_mat is not None, "Matrix a_mat must be provided."
        assert b is not None, "Vector b must be provided."
        assert a_mat.ndim == _EXPECTED_NDIM, (
            "a_mat is a matrix with shape (batch_size, n_constraints, dimension)."
        )
        assert b.ndim == _EXPECTED_NDIM, (
            "b is a matrix with shape (batch_size, n_constraints, 1)."
        )
        assert b.shape[2] == _LAST_AXIS_SIZE_B, (
            "b must have shape (batch_size, n_constraints, 1)."
        )
        # Batch broadcasting: either equal, or one side is 1.
        assert a_mat.shape[0] == b.shape[0] or a_mat.shape[0] == 1 or b.shape[0] == 1, (
            f"Batch sizes are inconsistent: a_mat{a_mat.shape}, b{b.shape}"
        )
        assert a_mat.shape[1] == b.shape[1], (
            "Number of rows in a_mat must equal size of b."
        )

        valid_methods = ["pinv", None]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method {method}. Valid methods are: {valid_methods}"
            )

        a_mat_pinv: BatchedEqPinv | None = None
        if method == "pinv" and not var_a_mat:
            a_mat_pinv = jnp.linalg.pinv(a_mat)

        self.a_mat = a_mat
        self.b = b
        self.a_mat_pinv = a_mat_pinv
        self.method = method
        self.var_b = var_b
        self.var_a_mat = var_a_mat

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
        a_mat = inp.eq.a_mat if inp.eq and self.var_a_mat else self.a_mat
        a_mat_pinv = inp.eq.a_mat_pinv if inp.eq and self.var_a_mat else self.a_mat_pinv
        return b, a_mat, a_mat_pinv

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project onto equality constraints.

        Args:
            yraw: ProjectionInstance to project.
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
            yraw: ProjectionInstance to project.
                The .x attribute is the point to project.

        Returns:
            The projected point for each point in the batch.
        """
        b, a_mat, a_mat_pinv = self.get_params(yraw)
        # a_mat must be available to apply the projection.
        assert a_mat is not None, (
            "a_mat must be provided in EqualityConstraintsSpecification "
            "when var_a_mat=True and a_mat_pinv is not supplied."
        )
        if a_mat_pinv is None:
            a_mat_pinv = jnp.linalg.pinv(a_mat)

        return yraw.update(x=yraw.x - a_mat_pinv @ (a_mat @ yraw.x - b))

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self.a_mat.shape[-1]

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.a_mat.shape[1]

    def cv(self, yraw: ProjectionInstance) -> BatchedScalar:
        """Compute the constraint violation.

        Args:
            yraw: ProjectionInstance to evaluate.

        Returns:
            The constraint violation for each point in the batch.
        """
        b, a_mat, _ = self.get_params(yraw)
        # a_mat must be available to compute the violation.
        assert a_mat is not None

        return jnp.linalg.norm(a_mat @ yraw.x - b, ord=jnp.inf, axis=1, keepdims=True)
