"""Cartesian constraint module for combining Box and SOC constraints."""

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from pinet._typing import BatchedScalar
from pinet.dataclasses import ProjectionInstance

from .base import Constraint
from .box import BoxConstraint
from .soc_constraint import SocConstraint


class CartesianConstraint(Constraint):
    """Cartesian product of Box and SOC constraints.

    This class combines multiple Box and SOC constraints that act on disjoint
    subsets of the variables (non-overlapping masks).

    Attributes:
        box_constraint: Optional box component of the Cartesian product.
        nl_constraints: SOC components.
        n_nonlinear: Number of non-linear (SOC) constituent constraints.
    """

    box_constraint: BoxConstraint | None
    nl_constraints: list[SocConstraint]
    _dim: int = eqx.field(static=True)
    n_nonlinear: int = eqx.field(static=True)

    def __init__(
        self,
        box_constraint: BoxConstraint | None = None,
        nl_constraints: list[SocConstraint] | None = None,
    ) -> None:
        """Initialize the Cartesian constraint.

        Args:
            box_constraint: Box constraint for the Cartesian product.
                Defaults to None.
            nl_constraints: List of non-linear constraints (SocConstraint).
                Defaults to None.

        Raises:
            ValueError: If no constraints are provided, if masks overlap,
                or if constraint dimensions are inconsistent.
        """
        nls: list[SocConstraint] = [] if nl_constraints is None else list(nl_constraints)

        self.box_constraint = box_constraint
        self.nl_constraints = nls
        self.n_nonlinear = len(nls)
        self._dim = self._validate_constraints(box_constraint, nls)

    @staticmethod
    def _validate_constraints(
        box_constraint: BoxConstraint | None,
        nl_constraints: list[SocConstraint],
    ) -> int:
        """Validate constraint types, dimensions, and mask overlap.

        Args:
            box_constraint: The optional box component.
            nl_constraints: The list of non-linear SOC components.

        Returns:
            The shared dimension across all constraints.

        Raises:
            ValueError: If no constraints are provided, if dimensions are
                inconsistent, or if masks overlap.
        """
        constraints: list[Constraint] = [
            c for c in (box_constraint, *nl_constraints) if c is not None
        ]
        if not constraints:
            raise ValueError("At least one constraint must be provided.")

        # The constituent constraint types (``BoxConstraint`` for the box
        # component, ``SocConstraint`` for the non-linear ones) are enforced by
        # the type system at construction time via the ``PINET_RUNTIME_CHECK``
        # beartype hook.

        # Validate that all constraints have the same dimension
        dim = constraints[0].dim
        for constraint in constraints:
            if constraint.dim != dim:
                raise ValueError(
                    f"All constraints must have the same dimension. "
                    f"Expected {dim}, got {constraint.dim}."
                )

        # Track which dimensions are already used. Masks depend only on
        # static shape information, so use ``numpy`` for the bookkeeping —
        # ``jnp.any`` would return a tracer when ``CartesianConstraint`` is
        # rebuilt inside the jitted ``solver.admm.initialize`` re-lift
        # (``var_a_mat=True`` path) and the ``if`` below would fail.
        used_mask = np.zeros(dim, dtype=bool)

        for constraint in constraints:
            if isinstance(constraint, BoxConstraint):
                # BoxConstraint.__init__ ensures mask is set.
                assert constraint.mask is not None
                new_mask = np.asarray(constraint.mask)
            else:
                # Narrow to SocConstraint so mask_u/mask_t are visible.
                assert isinstance(constraint, SocConstraint)
                new_mask = np.logical_or(
                    np.asarray(constraint.mask_u), np.asarray(constraint.mask_t)
                )
            if bool(np.any(np.logical_and(used_mask, new_mask))):
                raise ValueError(
                    "Constraint masks overlap with previously defined constraints."
                )
            used_mask = np.logical_or(used_mask, new_mask)

        return dim

    @property
    def constraints(self) -> list[Constraint]:
        """Return all constituent constraints in order."""
        return [c for c in (self.box_constraint, *self.nl_constraints) if c is not None]

    def project(self, y_raw: ProjectionInstance) -> ProjectionInstance:
        """Project the input to the feasible region.

        Projects onto each constraint independently. Since masks don't overlap,
        each constraint operates on a disjoint subset of variables.

        Args:
            y_raw: ProjectionInstance to project.

        Returns:
            The projected input.
        """
        if self.nl_constraints and not isinstance(y_raw.nl, (list, tuple)):
            raise TypeError(
                f"y_raw.nl must be a list or tuple, got {type(y_raw.nl).__name__}."
            )

        if self.box_constraint is not None:
            y_raw = self.box_constraint.project(y_raw)

        if self.nl_constraints:
            # Narrowed by the isinstance check above.
            assert y_raw.nl is not None
            for nl_constraint, nl_spec in zip(self.nl_constraints, y_raw.nl, strict=True):
                y_raw = y_raw.update(soc=nl_spec.to_primitive_spec())
                y_raw = nl_constraint.project(y_raw)

        return y_raw

    def cv(self, y_raw: ProjectionInstance) -> BatchedScalar:
        """Compute the constraint violation.

        Returns the maximum constraint violation across all constraints.

        Args:
            y_raw: ProjectionInstance to evaluate.

        Returns:
            The constraint violation for each point in the batch.
        """
        cvs = []

        if self.box_constraint is not None:
            cvs.append(self.box_constraint.cv(y_raw).reshape(-1))

        if self.nl_constraints:
            # cv requires per-instance SOC specs to be supplied on the input.
            assert y_raw.nl is not None, (
                "y_raw.nl must be provided when non-linear constraints are present."
            )
            for constraint, nl_spec in zip(self.nl_constraints, y_raw.nl, strict=True):
                y_raw = y_raw.update(soc=nl_spec.to_primitive_spec())
                cvs.append(constraint.cv(y_raw).reshape(-1))

        # Return the maximum violation
        if len(cvs) == 1:
            return cvs[0].reshape(-1, 1, 1)
        return jnp.max(jnp.array(cvs), 0).reshape(-1, 1, 1)

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the total number of constraints.

        Sums up the number of constraints from all constituent constraints.
        """
        total = 0
        for constraint in self.constraints:
            total += constraint.n_constraints
        return total
