"""Equality constraint module."""

import jax
import jax.numpy as jnp
from jax.experimental import checkify

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
        method: str = "pinv",
    ):
        """Initialize the equality constraint.

        Args:
            A: Left hand side matrix.
            b: Right hand side vector.
            method: A string that specifies the method used to solve
                linear systems. Valid method "pinv", "cholesky".
        """
        self.A = A
        self.b = b

        # Check if batch sizes are consistent.
        # They should either be the same, or one of them should be 1.
        checkify.check(
            self.A.shape[0] == self.b.shape[0]
            or self.A.shape[0] == 1
            or self.b.shape[0] == 1,
            f"Batch sizes are inconsistent: A{self.A.shape}, b{self.b.shape}",
        )

        checkify.check(
            self.A.shape[1] == b.shape[1], "Number of rows in A must equal size of b."
        )

        # List of valid methods
        valid_methods = ["pinv", "cholesky"]

        # TODO: Maybe include checks on if the chosen method
        # is applicable.
        # TODO: Add None as a method, to do no initialization.
        if method == "pinv":
            # Compute pseudo-inverse
            self.Apinv = jnp.linalg.pinv(self.A)
            # Instantiate projection method
            self.project = self.project_pinv
        elif method == "cholesky":
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

        else:
            raise ValueError(
                f"Invalid method {method}. Valid methods are: {valid_methods}"
            )

    def project_pinv(self, x: jnp.ndarray):
        """Project onto equality constraints using pseudo-inverse.

        Args:
            x: Point to be projected. Shape (n_batch, dimension, 1)
        """
        return x - self.Apinv @ (self.A @ x - self.b)

    def project_cholesky(self, x: jnp.ndarray):
        """Project onto equality contraints using cholesky factorization.

        Args:
            x: Point to be projected. Shape (n_batch, dimension, 1)
        """
        return x - jnp.matrix_transpose(self.A) @ self.cho_solve(self.A @ x - self.b)
