"""Box constraint module."""

from typing import Optional

from jax import numpy as jnp
from jax.experimental import checkify

from hcnn.constraints.base import Constraint


class BoxConstraint(Constraint):
    """Box constraint set.

    The box constraint set is defined as the Cartesian product of intervals.
    The interval is defined by a lower and an upper bound.
    The constraint possibly act only on a subset of the dimensions,
    defined by a mask.
    """

    def __init__(
        self,
        lower_bound: jnp.ndarray,
        upper_bound: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
    ):
        """Initialize the box constraint.

        Args:
            lower_bound: Lower bound of the box.
            upper_bound: Upper bound of the box.
            mask: Mask to apply the constraint only to some dimensions.
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if mask is None:
            mask = jnp.ones_like(lower_bound, dtype=jnp.bool_)
        self.mask = mask

        checkify.check(
            lower_bound.shape == upper_bound.shape,
            "Lower and upper bounds must have the same shape.",
        )
        checkify.check(
            lower_bound.shape[0] == jnp.sum(mask),
            "Number of active entries must be the same of the bounds.",
        )
        checkify.check(
            jnp.all(lower_bound <= upper_bound),
            "Lower bound must be less than or equal to the upper bound.",
        )
        checkify.check(mask.dtype == jnp.bool_, "Mask must be a boolean array.")

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """Project the input to the feasible region."""
        return x.at[self.mask].set(
            jnp.clip(x[self.mask], self.lower_bound, self.upper_bound)
        )
