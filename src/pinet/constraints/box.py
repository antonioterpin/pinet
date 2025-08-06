"""Box constraint module."""

from typing import Optional

import numpy as np
from jax import numpy as jnp
from jax.experimental.checkify import checkify

from pinet.dataclasses import ProjectionInstance

from .base import Constraint


class BoxConstraint(Constraint):
    """Box constraint set.

    The box constraint set is defined as the Cartesian product of intervals.
    The interval is defined by a lower and an upper bound.
    The constraint possibly acts only on a subset of the dimensions,
    defined by a mask.
    """

    lower_bound: jnp.ndarray
    upper_bound: jnp.ndarray

    def __init__(
        self,
        lower_bound: jnp.ndarray,
        upper_bound: jnp.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize the box constraint.

        Args:
            lower_bound (jnp.ndarray): Lower bound of the box.
                Shape (batch_size, n_constraints, 1).
            upper_bound (jnp.ndarray): Upper bound of the box.
                Shape (batch_size, n_constraints, 1).
            mask (jnp.ndarray): Mask to apply the constraint only to some dimensions.
                The same mask is applied to the entire batch.
                Must be a jnp.ndarray to be compatible with jit.
        """
        if mask is None:
            mask = np.ones(shape=(lower_bound.shape[1]), dtype=jnp.bool_)

        assert lower_bound is not None, "Lower bound must be provided."
        assert upper_bound is not None, "Upper bound must be provided."
        assert mask.dtype == jnp.bool_, "Mask must be a boolean array."
        assert (
            lower_bound.shape[1] == upper_bound.shape[1]
        ), "Lower and upper bounds must have the same shape."
        assert (
            lower_bound.shape[0] == upper_bound.shape[0]
            or lower_bound.shape[0] == 1
            or upper_bound.shape[0] == 1
        ), "Batch sizes of lower and upper bounds must be the same."
        checkify(
            (jnp.all(lower_bound <= upper_bound)),
            "Lower bound must be less than or equal to the upper bound.",
        )
        checkify(
            lower_bound.shape[1] == jnp.sum(mask),
            "Number of active entries must be the same of the bounds.",
        )

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.mask = mask
        self.masked_idx = tuple(np.where(mask)[0])

    def project(self, inp: ProjectionInstance) -> jnp.ndarray:
        """Project the input to the feasible region.

        Args:
            inp (ProjectionInstance): ProjectionInstance to projection.
                The .x attribute is the point to project.

        Returns:
            jnp.ndarray: The projected point for each point in the batch.
                Shape (batch_size, dimension, 1).
        """
        return inp.x.at[:, self.masked_idx, :].set(
            jnp.clip(inp.x[:, self.masked_idx, :], self.lower_bound, self.upper_bound)
        )

    def cv(self, inp: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            inp (ProjectionInstance): ProjectionInstance to evaluate.

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        tmp = jnp.maximum(
            jnp.max(inp.x[:, self.mask, :] - self.upper_bound, axis=1, keepdims=True),
            jnp.max(
                self.lower_bound - inp.x[:, self.masked_idx, :], axis=1, keepdims=True
            ),
        )
        return jnp.maximum(tmp, 0)

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self.lower_bound.shape[1]

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.lower_bound.shape[1]
