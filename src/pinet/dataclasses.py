"""This file contains dataclasses used to encapsulate inputs for the Pinet layer."""

from dataclasses import dataclass, replace
from typing import Optional

import jax
import jax.numpy as jnp


# Inputs dataclasses
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EqualityInputs:
    """Dataclass representing inputs used in forming equality constraints.

    Attributes:
        b (Optional[jnp.ndarray]): Vector representing the RHS of the equality constraint.
            Shape (batch_size, n_constraints, 1)
        A (Optional[jnp.ndarray]): Matrix representing the LHS of the equality constraint.
            Shape (batch_size, n_constraints, dimension).
        Apinv (Optional[jnp.ndarray]): The pseudoinverse of the matrix A.
            Shape (batch_size, dimension, n_constraints).
    """

    b: Optional[jnp.ndarray] = None
    A: Optional[jnp.ndarray] = None
    Apinv: Optional[jnp.ndarray] = None

    def update(self, **kwargs):
        """Update some attribute by keyword."""
        return replace(self, **kwargs)


# Inputs dataclasses
# TODO: Add dataclass for box constraints.
# TODO: Add dataclass for Inequality constraints.


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Inputs:
    """A dataclass for encapsulating model input parameters.

    Attributes:
        x (jnp.ndarray): The point to be projected.
            Shape (batch_size, dimension, 1)
        eq (EqualityInputs):
            An instance containing auxiliary inputs
            related to equality constraints.
    """

    x: jnp.ndarray
    eq: Optional[EqualityInputs] = EqualityInputs()

    def update(self, **kwargs):
        """Update some attribute by keyword."""
        return replace(self, **kwargs)
