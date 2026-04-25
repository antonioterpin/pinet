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
            TypeError: If nl_constraints is not a list or tuple.
        """
        if nl_constraints is None:
            nls: list[SocConstraint] = []
        elif isinstance(nl_constraints, (list, tuple)):
            nls = list(nl_constraints)
        else:
            raise TypeError(
                f"nl_constraints must be a list or tuple, "
                f"got {type(nl_constraints).__name__}."
            )

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
            ValueError: If no constraints are provided, if constraint types are
                invalid, if dimensions are inconsistent, or if masks overlap.
        """
        constraints: list[Constraint] = [
            c for c in (box_constraint, *nl_constraints) if c is not None
        ]
        if not constraints:
            raise ValueError("At least one constraint must be provided.")

        # Check that the constraints are boxes and socs
        if box_constraint is not None and not isinstance(box_constraint, BoxConstraint):
            raise ValueError(
                f"The box_constraint must be a BoxConstraint, "
                f"got {type(box_constraint).__name__}."
            )
        for constraint in nl_constraints:
            if not isinstance(constraint, SocConstraint):
                raise ValueError(
                    f"Only SocConstraint is currently supported "
                    f"in nl_constraints, got {type(constraint).__name__}."
                )

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
        # (``var_a_dyn=True`` path) and the ``if`` below would fail.
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

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project the input to the feasible region.

        Projects onto each constraint independently. Since masks don't overlap,
        each constraint operates on a disjoint subset of variables.

        Args:
            yraw: ProjectionInstance to project.

        Returns:
            The projected input.
        """
        if self.nl_constraints and not isinstance(yraw.nl, (list, tuple)):
            raise TypeError(
                f"yraw.nl must be a list or tuple, got {type(yraw.nl).__name__}."
            )

        if self.box_constraint is not None:
            yraw = self.box_constraint.project(yraw)

        if self.nl_constraints:
            # Narrowed by the isinstance check above.
            assert yraw.nl is not None
            for nl_constraint, nl_spec in zip(self.nl_constraints, yraw.nl, strict=True):
                yraw = yraw.update(soc=nl_spec.to_primitive_spec())
                yraw = nl_constraint.project(yraw)

        return yraw

    def cv(self, yraw: ProjectionInstance) -> BatchedScalar:
        """Compute the constraint violation.

        Returns the maximum constraint violation across all constraints.

        Args:
            yraw: ProjectionInstance to evaluate.

        Returns:
            The constraint violation for each point in the batch.
        """
        cvs = []

        if self.box_constraint is not None:
            cvs.append(self.box_constraint.cv(yraw).reshape(-1))

        if self.nl_constraints:
            # cv requires per-instance SOC specs to be supplied on the input.
            assert yraw.nl is not None, (
                "yraw.nl must be provided when non-linear constraints are present."
            )
            for constraint, nl_spec in zip(self.nl_constraints, yraw.nl, strict=True):
                yraw = yraw.update(soc=nl_spec.to_primitive_spec())
                cvs.append(constraint.cv(yraw).reshape(-1))

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
