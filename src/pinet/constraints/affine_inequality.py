"""Affine inequality constraint module."""

from jax import numpy as jnp

from pinet.dataclasses import ProjectionInstance

from .base import Constraint


class AffineInequalityConstraint(Constraint):
    """Affine inequality constraint set.

    The (affine) inequality constraint set is defined as:
    lb <= constr_matrix @ x <= ub
    where the constraint matrix and the vectors lb and ub are the parameters.
    """

    def __init__(
        self, constr_matrix: jnp.ndarray, lb: jnp.ndarray, ub: jnp.ndarray
    ) -> None:
        """Initialize the affine inequality constraint.

        Args:
            constr_matrix: The matrix in the inequality.
                Shape (batch_size, n_constraints, dimension).
            lb: The lower bound in the inequality.
                Shape (batch_size, n_constraints, 1).
            ub: The upper bound in the inequality.
                Shape (batch_size, n_constraints, 1).
        """
        self.constr_matrix = constr_matrix
        self.lb = lb
        self.ub = ub

        # Check if batch sizes for constr_matrix and l are consistent.
        # They should either be the same, or one of them should be 1.
        assert (
            self.constr_matrix.shape[0] == self.lb.shape[0]
            or self.constr_matrix.shape[0] == 1
            or self.lb.shape[0] == 1
        ), (
            "Batch sizes are inconsistent: "
            f"constr_matrix{self.constr_matrix.shape}, l{self.lb.shape}"
        )

        # Check if batch sizes for constr_matrix and u are consistent.
        # They should either be the same, or one of them should be 1.
        assert (
            self.constr_matrix.shape[0] == self.ub.shape[0]
            or self.constr_matrix.shape[0] == 1
            or self.ub.shape[0] == 1
        ), (
            "Batch sizes are inconsistent: "
            f"constr_matrix{self.constr_matrix.shape}, ub{self.ub.shape}"
        )

        assert self.constr_matrix.shape[1] == self.lb.shape[1], (
            "Number of rows in constr_matrix must equal size of l."
        )
        assert self.constr_matrix.shape[1] == self.ub.shape[1], (
            "Number of rows in constr_matrix must equal size of u."
        )

    def project(self, inp: ProjectionInstance) -> ProjectionInstance:
        """Project x onto the affine inequality constraint set.

        Args:
            inp: ProjectionInstance to projection.
                The .x attribute is the point to project.

        Returns:
            ProjectionInstance: The projected point for each point in the batch.
                Shape (batch_size, dimension, 1).

        Raises:
            NotImplementedError: Always. This method must not be called directly.
        """
        raise NotImplementedError(
            "The 'project' method is not implemented and should not be called."
        )

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self.constr_matrix.shape[-1]

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.constr_matrix.shape[1]

    def cv(self, inp: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            inp: ProjectionInstance to evaluate.

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        constr_matrix_x = self.constr_matrix @ inp.x
        cv_ub = jnp.maximum(constr_matrix_x - self.ub, 0)
        cv_lb = jnp.maximum(self.lb - constr_matrix_x, 0)
        return jnp.max(jnp.maximum(cv_ub, cv_lb), axis=1, keepdims=True)
