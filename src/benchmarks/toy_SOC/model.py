"""Module for setting up Pinet models for toy SOC."""

from typing import Callable

import jax.numpy as jnp
from flax import linen as nn


class HardConstrainedMLP(nn.Module):
    """A simple MLP model for solving the hard constrained problem."""

    activation: nn.Module
    layers: list[int]
    project: Callable[
        [jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    m: int
    n: int

    @nn.compact
    def __call__(
        self,
        input: dict[str, jnp.ndarray],
    ):
        """Call the NN.

        Args:
            input (dict):
                Dictionary containing the input data with keys "b" and "c".

        Returns:
            jnp.ndarray:
                Output of the MLP, projected onto the feasible set.
        """
        b, c = input["b"].squeeze(-1), input["c"].squeeze(-1)
        x = jnp.concatenate((b, c), axis=-1)
        for layer_size in self.layers:
            x = self.activation(nn.Dense(layer_size)(x))
        # Final layer to project
        x = nn.Dense(self.n + self.m)(x).reshape((x.shape[0], self.n + self.m, 1))
        x = self.project(jnp.zeros_like(x), x, b.reshape((b.shape[0], -1, 1)))[0]
        return x
