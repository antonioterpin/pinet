"""Flax implementation of the projection layer."""

import jax
from flax import linen as nn
from jax import numpy as jnp


class Project(nn.Module):
    """Projection layer implemented via iterative projections."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Project the input to the feasible region."""
        jax.debug.print("Projection layer not implemented. No projection is performed.")
        return x
