"""Box constraint module."""

from typing import Optional

from jax import numpy as jnp

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
                The same mask is applied to the entire batch.
                TODO: Make the mask batch-dependent?
        """
        assert lower_bound is not None, "Lower bound must be provided."
        assert upper_bound is not None, "Upper bound must be provided."
        assert (
            lower_bound.shape[1] == upper_bound.shape[1]
        ), "Lower and upper bounds must have the same shape."
        # Check batch size consistency
        assert (
            lower_bound.shape[0] == upper_bound.shape[0]
            or lower_bound.shape[0] == 1
            or upper_bound.shape[0] == 1
        ), "Batch sizes of lower and upper bounds must be the same."
        assert mask is None or lower_bound.shape[1] == jnp.sum(
            mask
        ), "Number of active entries must be the same of the bounds."
        assert jnp.all(
            lower_bound <= upper_bound
        ), "Lower bound must be less than or equal to the upper bound."
        assert mask is None or mask.dtype == jnp.bool_, "Mask must be a boolean array."

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if mask is None:
            mask = jnp.ones(shape=(self.lower_bound.shape[1]), dtype=jnp.bool_)
        self.mask = mask

    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """Project the input to the feasible region.

        Args:
            x: Input to be projected. jnp.ndarray with shape (n, d).
        """
        return x.at[:, self.mask].set(
            jnp.clip(x[:, self.mask], self.lower_bound, self.upper_bound)
        )

    @property
    def dim(self):
        """Return the dimension of the constraint set."""
        return self.mask.shape[-1]

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.lower_bound.shape[1]
