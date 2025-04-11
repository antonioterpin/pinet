"""Abstract class for constraint sets."""

from abc import abstractmethod

import jax.numpy as jnp


class Constraint:
    """Abstract class for constraint sets."""

    @abstractmethod
    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """Project the input to the feasible region."""
        pass

    @abstractmethod
    def cv(self, x: jnp.ndarray) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            x (jnp.ndarray): Point to be evaluated. Shape (batch_size, dimension, 1).

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        pass

    @property
    @abstractmethod
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        pass
