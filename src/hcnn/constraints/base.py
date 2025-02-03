"""Abstract class for constraint sets."""

from abc import abstractmethod

import jax.numpy as jnp


class Constraint:
    """Abstract class for constraint sets."""

    @abstractmethod
    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """Project the input to the feasible region."""
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
