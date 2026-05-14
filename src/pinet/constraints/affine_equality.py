"""Equality constraint module."""

import jax.numpy as jnp

from pinet.dataclasses import ProjectionInstance

from .base import Constraint


class EqualityConstraint(Constraint):
    """Equality constraint set.

    The (affine) equality constraint set is defined as:
    a @ x == b
    where the matrix a and the vector b are the parameters.
    It might be worth to consider masking, so that the
    constraint acts only on a subset of dimensions.
    """

    def __init__(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        method: str | None = "pinv",
        var_b: bool | None = False,
        var_a: bool | None = False,
    ) -> None:
        """Initialize the equality constraint.

        Args:
            a: Left hand side matrix.
                Shape (batch_size, n_constraints, dimension).
            b: Right hand side vector.
                Shape (batch_size, n_constraints, 1).
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

        self.a = a
        self.b = b
        self.method = method
        self.var_b = var_b
        self.var_a = var_a

        self.EXPECTED_NDIM = 3
        self.LAST_AXIS_SIZE_B = 1

        self.setup()

    def setup(self) -> None:
        """Sets up the equality constraint.

        Raises:
            Exception: If the provided method is not valid.
        """

        # The equality matrix is batched as (batch_size, n_constraints, dimension).
        assert self.a.ndim == self.EXPECTED_NDIM, (
            "a is a matrix with shape (batch_size, n_constraints, dimension)."
        )
        # The right-hand side is batched with the same rank as a.
        assert self.b.ndim == self.EXPECTED_NDIM, (
            "b is a matrix with shape (batch_size, n_constraints, 1)."
        )
        # The last axis of b stores a single scalar per constraint.
        assert self.b.shape[2] == self.LAST_AXIS_SIZE_B, (
            "b must have shape (batch_size, n_constraints, 1)."
        )

        # Check if batch sizes are consistent.
        # They should either be the same, or one of them should be 1.
        # a and b must be batch-compatible, allowing singleton broadcasting.
        assert (
            self.a.shape[0] == self.b.shape[0]
            or self.a.shape[0] == 1
            or self.b.shape[0] == 1
        ), f"Batch sizes are inconsistent: a{self.a.shape}, b{self.b.shape}"

        # Each equality row in a needs one matching entry in b.
        assert self.a.shape[1] == self.b.shape[1], (
            "Number of rows in a must equal size of b."
        )

        # List of valid methods
        valid_methods = ["pinv", None]

        if self.method == "pinv":
            if not self.var_a:
                self.a_pinv = jnp.linalg.pinv(self.a)

            self.project = self.project_pinv
        elif self.method is None:

            def raise_not_implemented_error(
                yraw: ProjectionInstance,
            ) -> ProjectionInstance:
                raise NotImplementedError("No projection method set.")

            self.project = raise_not_implemented_error
        else:
            raise Exception(
                f"Invalid method {self.method}. Valid methods are: {valid_methods}"
            )

    def get_params(
        self, inp: ProjectionInstance
    ) -> tuple[jnp.ndarray, jnp.ndarray | None, jnp.ndarray | None]:
        """Get matrix, b, matrix_pinv depending on varying constraints.

        Args:
            inp: ProjectionInstance to get parameters from.

        Returns:
            tuple Tuple containing
                b: Right hand side vector.
                    Shape (batch_size, n_constraints, 1).
                matrix: Left hand side matrix.
                    Shape (batch_size, n_constraints, dimension).
                matrix_pinv: Pseudo-inverse of matrix.
                    Shape (batch_size, n_constraints, dimension).
        """
        b = inp.eq.b if inp.eq and inp.eq.b is not None else self.b
        a = inp.eq.a if inp.eq and self.var_a else self.a
        a_pinv = inp.eq.a_pinv if inp.eq and self.var_a else self.a_pinv

        return b, a, a_pinv

    def project_pinv(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project onto equality constraints using pseudo-inverse.

        Args:
            yraw: ProjectionInstance to projection.
                The .x attribute is the point to project.

        Returns:
            ProjectionInstance: The projected point for each point in the batch.
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

    def cv(self, yraw: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            yraw: ProjectionInstance to evaluate.

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape    (batch_size, 1, 1).
        """
        b, a, _ = self.get_params(yraw)
        # a must be available to compute the violation.
        assert a is not None

        return jnp.linalg.norm(a @ yraw.x - b, ord=jnp.inf, axis=1, keepdims=True)
