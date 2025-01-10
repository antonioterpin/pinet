"""Abstract class for constraint sets."""

from abc import abstractmethod

import jax.numpy as jnp


class Constraint:
    """Abstract class for constraint sets."""

    @abstractmethod
    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """Project the input to the feasible region."""
        pass
