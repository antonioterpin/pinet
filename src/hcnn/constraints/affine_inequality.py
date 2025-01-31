"""Affine inequality constraint module."""

from jax.experimental import checkify

from hcnn.constraints.base import Constraint


class AffineInequalityConstraint(Constraint):
    """Affine inequality constraint set.

    The (affine) inequality constraint set is defined as:
    l <= C @ x <= u
    where the matrix C and the vectors l and u are the parameters.
    """

    def __init__(self, C, lb, ub):
        """Initialize the affine inequality constraint.

        Args:
            C (numpy.ndarray): The matrix C in the inequality.
            lb (numpy.ndarray): The lower bound in the inequality.
            ub (numpy.ndarray): The upper bound in the inequality.
        """
        self.C = C
        self.lb = lb
        self.ub = ub

        # Check if batch sizes for C and l are consistent.
        # They should either be the same, or one of them should be 1.
        checkify.check(
            self.C.shape[0] == self.lb.shape[0]
            or self.C.shape[0] == 1
            or self.lb.shape[0] == 1,
            f"Batch sizes are inconsistent: C{self.C.shape}, l{self.lb.shape}",
        )

        # Check if batch sizes for C and u are consistent.
        # They should either be the same, or one of them should be 1.
        checkify.check(
            self.C.shape[0] == self.ub.shape[0]
            or self.C.shape[0] == 1
            or self.ub.shape[0] == 1,
            f"Batch sizes are inconsistent: C{self.C.shape}, ub{self.ub.shape}",
        )

        checkify.check(
            self.C.shape[1] == self.lb.shape[1],
            "Number of rows in C must equal size of l.",
        )
        checkify.check(
            self.C.shape[1] == self.ub.shape[1],
            "Number of rows in C must equal size of u.",
        )

    def project(self, x):
        """Project x onto the affine inequality constraint set.

        Args:
            x (numpy.ndarray): The point to be projected. Shape (batch_size, n_dims).
        """
        raise NotImplementedError(
            "The 'project' method is not implemented and should not be called."
        )

    @property
    def dim(self):
        """Return the dimension of the constraint set."""
        return self.C.shape[-1]

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.C.shape[1]
