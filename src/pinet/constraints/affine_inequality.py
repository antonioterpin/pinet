"""Affine inequality constraint module."""

from jax import numpy as jnp

from pinet._typing import BatchedIneqBound, BatchedIneqMatrix, BatchedScalar
from pinet.dataclasses import ProjectionInstance

from .base import Constraint


class AffineInequalityConstraint(Constraint):
    """Affine inequality constraint set.

    The (affine) inequality constraint set is defined as:
    lb <= c_mat @ x <= ub
    where the constraint matrix and the vectors lb and ub are the parameters.

    Attributes:
        c_mat: Matrix in the inequality.
        lb: Lower bound.
        ub: Upper bound.
    """

    c_mat: BatchedIneqMatrix
    lb: BatchedIneqBound
    ub: BatchedIneqBound

    def __init__(
        self,
        c_mat: BatchedIneqMatrix,
        lb: BatchedIneqBound,
        ub: BatchedIneqBound,
    ) -> None:
        """Initialize the affine inequality constraint.

        Args:
            c_mat: The matrix in the inequality.
            lb: The lower bound in the inequality.
            ub: The upper bound in the inequality.
        """
        # Batch broadcasting: either equal, or one side is 1.
        assert c_mat.shape[0] == lb.shape[0] or c_mat.shape[0] == 1 or lb.shape[0] == 1, (
            f"Batch sizes are inconsistent: c_mat{c_mat.shape}, l{lb.shape}"
        )
        assert c_mat.shape[0] == ub.shape[0] or c_mat.shape[0] == 1 or ub.shape[0] == 1, (
            f"Batch sizes are inconsistent: c_mat{c_mat.shape}, ub{ub.shape}"
        )
        assert c_mat.shape[1] == lb.shape[1], (
            "Number of rows in c_mat must equal size of l."
        )
        assert c_mat.shape[1] == ub.shape[1], (
            "Number of rows in c_mat must equal size of u."
        )

        self.c_mat = c_mat
        self.lb = lb
        self.ub = ub

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project x onto the affine inequality constraint set.

        Args:
            yraw: ProjectionInstance to project.
                The .x attribute is the point to project.

        Returns:
            The projected point for each point in the batch.

        Raises:
            NotImplementedError: Always. This method must not be called directly.
        """
        raise NotImplementedError(
            "The 'project' method is not implemented and should not be called."
        )

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self.c_mat.shape[-1]

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.c_mat.shape[1]

    def cv(self, yraw: ProjectionInstance) -> BatchedScalar:
        """Compute the constraint violation.

        Args:
            yraw: ProjectionInstance to evaluate.

        Returns:
            The constraint violation for each point in the batch.
        """
        c_mat_x = self.c_mat @ yraw.x
        cv_ub = jnp.maximum(c_mat_x - self.ub, 0)
        cv_lb = jnp.maximum(self.lb - c_mat_x, 0)
        return jnp.max(jnp.maximum(cv_ub, cv_lb), axis=1, keepdims=True)
