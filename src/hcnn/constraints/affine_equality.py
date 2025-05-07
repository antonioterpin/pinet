"""Equality constraint module."""

from typing import Optional

import jax
import jax.numpy as jnp

from hcnn.constraints.base import Constraint


class EqualityConstraint(Constraint):
    """Equality constraint set.

    The (affine) equality constraint set is defined as:
    A @ x == b
    where the matrix A and the vector b are the parameters.
    It might be worth to consider masking, so that the
    constraint acts only on a subset of dimensions.
    """

    def __init__(
        self,
        A: jnp.ndarray,
        b: jnp.ndarray,
        method: Optional[str] = "pinv",
        var_b: Optional[bool] = False,
        var_A: Optional[bool] = False,
    ):
        """Initialize the equality constraint.

        Args:
            A: Left hand side matrix.
            b: Right hand side vector.
            method: A string that specifies the method used to solve
                linear systems. Valid method "pinv", "cholesky", None.
            var_b: Boolean that indicates whether the b vector
                changes or is constant.
            var_A: Boolean that indicates whether the A matrix
                changes or is constant.
        """
        assert A is not None, "Matrix A must be provided."
        assert b is not None, "Vector b must be provided."

        self.A = A
        self.b = b
        self.method = method
        self.var_b = var_b
        self.var_A = var_A

        self.setup()

    def setup(self):
        """Sets up the equality constraint."""
        assert (
            self.A.ndim == 3
        ), "A is a matrix with shape (n_batch, n_constraints, dimension)."
        assert self.b.ndim == 3, "b is a matrix with shape (n_batch, n_constraints, 1)."
        assert self.b.shape[2] == 1, "b must have shape (n_batch, n_constraints, 1)."

        # Check if batch sizes are consistent.
        # They should either be the same, or one of them should be 1.
        assert (
            self.A.shape[0] == self.b.shape[0]
            or self.A.shape[0] == 1
            or self.b.shape[0] == 1
        ), f"Batch sizes are inconsistent: A{self.A.shape}, b{self.b.shape}"

        assert (
            self.A.shape[1] == self.b.shape[1]
        ), "Number of rows in A must equal size of b."

        # List of valid methods
        valid_methods = ["pinv", "cholesky", None]

        # TODO: Maybe include checks on if the chosen method
        # is applicable.
        if self.method == "pinv":
            if self.var_A:
                self.project = self.project_pinv_vAb
            else:
                # Compute pseudo-inverse
                self.Apinv = jnp.linalg.pinv(self.A)
                # Instantiate projection method
                if self.var_b:
                    self.project = self.project_pinv_vb
                else:
                    self.project = self.project_pinv
        # TODO: Implement cholesky projection methods for variable A, b.
        elif self.method == "cholesky":
            # Compute gramian of A
            Agram = self.A @ jnp.matrix_transpose(self.A)
            # Compute Cholesky factorization (pretty efficient for PSD matrices)
            cfac = jax.scipy.linalg.cho_factor(Agram, lower=False)
            # Handling of batch dimension
            if self.A.shape[0] == 1:
                self.cho_solve = jax.vmap(
                    lambda x: jax.scipy.linalg.cho_solve((cfac[0][0, :, :], False), x)
                )
            else:
                self.cho_solve = lambda x: jax.scipy.linalg.cho_solve(cfac, x)
            # Instantiate projection method
            self.project = self.project_cholesky
        elif self.method is None:

            def raise_not_implemented_error():
                raise NotImplementedError("No projection method set.")

            self.project = lambda _: raise_not_implemented_error()
        else:
            raise Exception(
                f"Invalid method {self.method}. Valid methods are: {valid_methods}"
            )

    def project_pinv(self, x: jnp.ndarray):
        """Project onto equality constraints using pseudo-inverse.

        Args:
            x: Point to be projected. Shape (n_batch, dimension, 1)
        """
        return x - self.Apinv @ (self.A @ x - self.b)

    def project_pinv_vb(self, x: jnp.ndarray, b: jnp.ndarray):
        """Project onto equality constraints using pseudo-inverse.

        Args:
            x: Point to be projected. Shape (n_batch, dimension, 1)
            b: Right hand side vector.
        """
        return x - self.Apinv @ (self.A @ x - b)

    def project_pinv_vAb(
        self,
        x: jnp.ndarray,
        b: jnp.ndarray,
        A: jnp.ndarray,
        Apinv: jnp.ndarray,
    ):
        """Project onto equality constraints using pseudo-inverse.

        Args:
            x: Point to be projected. Shape (n_batch, dimension, 1)
            b: Right hand side vector.
            A: Left hand side matrix.
            Apinv: Pseudo-inverse of A.
        """
        return x - Apinv @ (A @ x - b)

    def project_cholesky(self, x: jnp.ndarray):
        """Project onto equality contraints using cholesky factorization.

        Args:
            x: Point to be projected. Shape (n_batch, dimension, 1)
        """
        return x - jnp.matrix_transpose(self.A) @ self.cho_solve(self.A @ x - self.b)

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self.A.shape[-1]

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.A.shape[1]

    def cv(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            x: Point to be evaluated. Shape (batch_size, dimension, 1).

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        return jnp.linalg.norm(self.A @ x - self.b, ord=jnp.inf, axis=1, keepdims=True)
