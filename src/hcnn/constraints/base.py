"""Abstract class for constraint sets."""

from abc import abstractmethod

import jax.numpy as jnp

from hcnn.utils import Inputs


class Constraint:
    """Abstract class for constraint sets."""

    @abstractmethod
    def project(self, x: Inputs) -> jnp.ndarray:
        """Project the input to the feasible region."""
        pass

    @abstractmethod
    def cv(self, inp: Inputs) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            inp (Inputs): Inputs to evaluate.

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
