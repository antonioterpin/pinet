"""Flax implementation of the projection layer."""

from typing import Callable, List

from flax import linen as nn
from jax import numpy as jnp

from hcnn.constraints.base import Constraint


class Project(nn.Module):
    """Projection layer implemented via iterative projections."""

    constraints: List[Constraint]
    schedule: Callable[[int], float]

    @nn.compact
    def __call__(self, x: jnp.ndarray, step: int = 0) -> jnp.ndarray:
        """Project the input to the feasible region."""
        y = x
        y = jnp.expand_dims(y, axis=2)
        for constraint in self.constraints:
            y = constraint.project(y)
        interpolation_value = self.schedule(step)
        y = y.reshape(x.shape)
        return interpolation_value * x + (1 - interpolation_value) * y
