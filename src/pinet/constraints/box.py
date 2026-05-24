"""Box constraint module."""

import equinox as eqx
import numpy as np
from jax import numpy as jnp

from pinet._typing import BatchedRHS, BatchedScalar, ColScaling, Mask1D
from pinet.dataclasses import BoxConstraintSpecification, ProjectionInstance

from .base import Constraint


class BoxConstraint(Constraint):
    """Box constraint set.

    The box constraint set is defined as the Cartesian product of intervals.
    The interval is defined by a lower and an upper bound.
    The constraint possibly acts only on a subset of the dimensions,
    defined by a mask.

    Attributes:
        lb: Lower bound for each constrained coordinate.
        ub: Upper bound for each constrained coordinate.
        mask: Boolean mask selecting which dimensions are constrained.
        scale: Scaling vector applied to variable bounds passed via ``y_raw.box``.
    """

    lb: BatchedRHS | None
    ub: BatchedRHS | None
    mask: Mask1D
    scale: ColScaling
    _dim: int = eqx.field(static=True)
    _n_constraints: int = eqx.field(static=True)

    def __init__(
        self,
        box_spec: BoxConstraintSpecification,
        scale: ColScaling | None = None,
    ) -> None:
        """Initialize the box constraint.

        Args:
            box_spec: Specification of the box constraint.
                For variable bounds, provide an example of the bounds.
            scale: Optional scaling vector applied to variable bounds passed via
                ``y_raw.box``. Defaults to all ones.
        """
        lb = box_spec.lb
        ub = box_spec.ub
        mask = box_spec.mask
        if mask is not None:
            dim = int(mask.size)
        elif lb is not None:
            dim = lb.shape[1]
        else:
            # Either lb or ub must be provided if mask is not set.
            assert ub is not None
            dim = ub.shape[1]

        if lb is not None:
            n_constraints = lb.shape[1]
        else:
            # At least one of lb or ub must be provided.
            assert ub is not None
            n_constraints = ub.shape[1]

        self.lb = lb
        self.ub = ub
        self.mask = (
            mask
            if mask is not None
            else jnp.asarray(np.ones(shape=(dim,), dtype=jnp.bool_))
        )
        # Default scale is applied to per-instance bounds supplied via
        # ``y_raw.box.lb/ub``, which have shape ``(batch, n_constraints, 1)``.
        # Using ``dim`` here would broadcast-error whenever ``mask`` is set
        # (``dim != n_constraints``).
        self.scale = scale if scale is not None else jnp.ones((1, n_constraints, 1))
        self._dim = dim
        self._n_constraints = n_constraints

    def get_params(
        self, y_raw: ProjectionInstance
    ) -> tuple[BatchedRHS, BatchedRHS, Mask1D]:
        """Get the parameters of the box constraint.

        Args:
            y_raw: ProjectionInstance to get the parameters from.

        Returns:
            A tuple ``(lb, ub, mask)`` with the lower/upper bounds and the mask.
        """
        lb = (
            (y_raw.box.lb * self.scale)
            if y_raw.box and y_raw.box.lb is not None
            else self.lb
        )
        ub = (
            (y_raw.box.ub * self.scale)
            if y_raw.box and y_raw.box.ub is not None
            else self.ub
        )
        mask = y_raw.box.mask if y_raw.box and y_raw.box.mask is not None else self.mask
        if lb is None:
            # An absent lower bound is treated as an unbounded interval below.
            assert ub is not None
            lb = -jnp.inf * jnp.ones_like(ub)
        if ub is None:
            ub = jnp.inf * jnp.ones_like(lb)
        # A mask is always required to know which coordinates are clipped.
        assert mask is not None

        return lb, ub, mask

    def project(self, y_raw: ProjectionInstance) -> ProjectionInstance:
        """Project the input to the feasible region.

        Args:
            y_raw: ProjectionInstance to project.
                The .x attribute is the point to project.

        Returns:
            The projected point for each point in the batch.
        """
        lb, ub, mask = self.get_params(y_raw)
        # ``y_raw.x`` is annotated ``ArrayLike`` (jax.Array | np.ndarray) at the
        # public API boundary; ``.at`` is a JAX-only construct, so coerce
        # before slicing.
        x = jnp.asarray(y_raw.x)
        return y_raw.update(x=x.at[:, mask, :].set(jnp.clip(x[:, mask, :], lb, ub)))

    def cv(self, y_raw: ProjectionInstance) -> BatchedScalar:
        """Compute the constraint violation.

        Args:
            y_raw: ProjectionInstance to evaluate.

        Returns:
            The constraint violation for each point in the batch.
        """
        lb, ub, mask = self.get_params(y_raw)
        cvs = jnp.maximum(
            jnp.max(y_raw.x[:, mask, :] - ub, axis=1, keepdims=True),
            jnp.max(lb - y_raw.x[:, mask, :], axis=1, keepdims=True),
        )
        return jnp.maximum(cvs, 0)

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self._n_constraints
