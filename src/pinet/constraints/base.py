"""Abstract class for constraint sets."""

from abc import abstractmethod

import jax.numpy as jnp

from pinet.dataclasses import ProjectionInstance


class Constraint:
    """Abstract class for constraint sets."""

    @abstractmethod
    def project(self, x: ProjectionInstance) -> jnp.ndarray:
        """Project the input to the feasible region."""
        pass

    @abstractmethod
    def cv(self, inp: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            inp (ProjectionInstance): ProjectionInstance to evaluate.

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
