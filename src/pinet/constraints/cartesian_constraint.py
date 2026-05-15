"""Cartesian constraint module for combining Box and SOC constraints."""

import jax.numpy as jnp

from pinet.dataclasses import ProjectionInstance, SocConstraintSpecification

from .base import Constraint
from .box import BoxConstraint
from .soc_constraint import SocConstraint


class CartesianConstraint(Constraint):
    """Cartesian product of Box and SOC constraints.

    This class combines multiple Box and SOC constraints that act on disjoint
    subsets of the variables (non-overlapping masks).
    """

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
        self.box_constraint = box_constraint

        if nl_constraints is None:
            self.nl_constraints = []
        elif isinstance(nl_constraints, (list, tuple)):
            self.nl_constraints = list(nl_constraints)
        else:
            raise TypeError(
                f"nl_constraints must be a list or tuple, "
                f"got {type(nl_constraints).__name__}."
            )

        self.constraints = [
            c for c in (box_constraint, *self.nl_constraints) if c is not None
        ]

        self._validate_constraints()

        self.n_nonlinear = len(self.nl_constraints)

        # These update functions should not change
        # the nl attribute of yraw!
        self.update_fns = []

        # Define update functions
        def soc_update(
            yraw: ProjectionInstance, socspec: SocConstraintSpecification
        ) -> ProjectionInstance:
            return yraw.update(soc=socspec)

        for _ in self.nl_constraints:
            # Placeholder for more constraints
            # if isinstance(constraint, SocConstraint):
            self.update_fns.append(soc_update)

    def _validate_constraints(self) -> None:
        """Validate constraint types, dimensions, and mask overlap.

        Raises:
            ValueError: If no constraints are provided, if constraint types are
                invalid, if dimensions are inconsistent, or if masks overlap.
        """
        if not self.constraints:
            raise ValueError("At least one constraint must be provided.")

        # Check that the constraints are boxes and socs
        if self.box_constraint is not None and not isinstance(
            self.box_constraint, BoxConstraint
        ):
            raise ValueError(
                f"The box_constraint must be a BoxConstraint, "
                f"got {type(self.box_constraint).__name__}."
            )
        for constraint in self.nl_constraints:
            if not isinstance(constraint, SocConstraint):
                raise ValueError(
                    f"Only SocConstraint is currently supported "
                    f"in nl_constraints, got {type(constraint).__name__}."
                )

        # Validate that all constraints have the same dimension
        self._dim = self.constraints[0].dim
        for constraint in self.constraints:
            if constraint.dim != self._dim:
                raise ValueError(
                    f"All constraints must have the same dimension. "
                    f"Expected {self._dim}, got {constraint.dim}."
                )

        # Create a mask to track which dimensions are already used
        used_mask = jnp.zeros(self._dim, dtype=bool)

        for constraint in self.constraints:
            if isinstance(constraint, BoxConstraint):
                # BoxConstraint.__init__ ensures mask is set.
                assert constraint.mask is not None
                new_mask = jnp.asarray(constraint.mask)
            else:
                # SOC constraints have both mask_u and mask_t
                new_mask = jnp.logical_or(constraint.mask_u, constraint.mask_t)
            if jnp.any(jnp.logical_and(used_mask, new_mask)):
                raise ValueError(
                    "Constraint masks overlap with previously defined constraints."
                )
            used_mask = jnp.logical_or(used_mask, new_mask)

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project the input to the feasible region.

        Projects onto each constraint independently. Since masks don't overlap,
        each constraint operates on a disjoint subset of variables.

        Args:
            yraw: ProjectionInstance to project.

        Returns:
            ProjectionInstance: The projected input.
        """
        if self.nl_constraints and not isinstance(yraw.nl, (list, tuple)):
            raise TypeError(
                f"yraw.nl must be a list or tuple, got {type(yraw.nl).__name__}."
            )

        if self.box_constraint is not None:
            yraw = self.box_constraint.project(yraw)

        # Project onto each constraint
        if self.nl_constraints:
            # Narrowed by the isinstance check above.
            assert yraw.nl is not None
            for nl_constraint, update_fn, nl_spec in zip(
                self.nl_constraints, self.update_fns, yraw.nl, strict=True
            ):
                yraw = update_fn(yraw, nl_spec.to_primitive_spec())
                yraw = nl_constraint.project(yraw)

        return yraw

    def cv(self, yraw: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Returns the maximum constraint violation across all constraints.

        Args:
            yraw: ProjectionInstance to evaluate.

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        cvs = []

        if self.box_constraint is not None:
            cvs.append(self.box_constraint.cv(yraw).reshape(-1))

        # Compute constraint violations for nl constraints
        if self.nl_constraints:
            # cv requires per-instance SOC specs to be supplied on the input.
            assert yraw.nl is not None, (
                "yraw.nl must be provided when non-linear constraints are present."
            )
            for constraint, update_fn, nl_spec in zip(
                self.nl_constraints, self.update_fns, yraw.nl, strict=True
            ):
                yraw = update_fn(yraw, nl_spec.to_primitive_spec())
                cvs.append(constraint.cv(yraw).reshape(-1))

        # Return the maximum violation
        if len(cvs) == 1:
            return cvs[0].reshape(-1, 1, 1)
        else:
            return jnp.max(jnp.array(cvs), 0).reshape(-1, 1, 1)

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set.

        Returns:
            int: The dimension of the constraint set.
        """
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the total number of constraints.

        Sums up the number of constraints from all constituent constraints.

        Returns:
            int: The total number of constraints.
        """
        total = 0
        for constraint in self.constraints:
            total += constraint.n_constraints
        return total
