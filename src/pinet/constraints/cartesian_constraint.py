"""Cartesian constraint module for combining Box and SOC constraints."""

from typing import List, Optional

import jax.numpy as jnp

from pinet.constraints.base import Constraint
from pinet.constraints.box import BoxConstraint
from pinet.constraints.soc_constraint import SocConstraint
from pinet.dataclasses import ProjectionInstance


class CartesianConstraint(Constraint):
    """Cartesian product of Box and SOC constraints.

    This class combines multiple Box and SOC constraints that act on disjoint
    subsets of the variables (non-overlapping masks).
    """

    def __init__(
        self,
        constraints: Optional[List[Constraint]] = None,
    ) -> None:
        """Initialize the Cartesian constraint.

        Args:
            constraints (Optional[List[Constraint]]):
                List of constraints (BoxConstraint or SocConstraint).

        Raises:
            ValueError: If no constraints are provided or if masks overlap.
        """
        self.constraints = constraints or []

        if not self.constraints:
            raise ValueError("At least one constraint must be provided.")

        # Get dimension from the first constraint
        self._dim = self.constraints[0].dim

        # Check that the constraints are boxes and socs
        for constraint in self.constraints:
            if not isinstance(constraint, (BoxConstraint, SocConstraint)):
                raise ValueError(
                    "Only BoxConstraint and SocConstraint are supported "
                    "in CartesianConstraint."
                )

        # Validate that all constraints have the same dimension
        for constraint in self.constraints:
            if constraint.dim != self._dim:
                raise ValueError(
                    f"All constraints must have the same dimension. "
                    f"Expected {self._dim}, got {constraint.dim}."
                )

        # Check that masks don't overlap
        self._validate_masks()

    def _validate_masks(self) -> None:
        """Validate that masks from all constraints do not overlap.

        Raises:
            ValueError: If any masks overlap.
        """
        # Create a mask to track which dimensions are already used
        used_mask = jnp.zeros(self._dim, dtype=bool)

        for constraint in self.constraints:
            if isinstance(constraint, BoxConstraint):
                if jnp.any(jnp.logical_and(used_mask, constraint.mask)):
                    raise ValueError(
                        "Constraint masks overlap with previously defined constraints."
                    )
                used_mask = jnp.logical_or(constraint.mask, used_mask)
            elif isinstance(constraint, SocConstraint):
                # SOC constraints have both mask_u and mask_t
                soc_mask = jnp.logical_or(constraint.mask_u, constraint.mask_t)
                if jnp.any(jnp.logical_and(used_mask, soc_mask)):
                    raise ValueError(
                        "Constraint masks overlap with previously defined constraints."
                    )
                used_mask = jnp.logical_or(used_mask, soc_mask)

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project the input to the feasible region.

        Projects onto each constraint independently. Since masks don't overlap,
        each constraint operates on a disjoint subset of variables.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to project.

        Returns:
            ProjectionInstance: The projected input.
        """
        result = yraw

        # Project onto each constraint
        for constraint in self.constraints:
            result = constraint.project(result)

        return result

    def cv(self, yraw: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Returns the maximum constraint violation across all constraints.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to evaluate.

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        cvs = []

        # Compute constraint violations for all constraints
        for constraint in self.constraints:
            cvs.append(constraint.cv(yraw).reshape(-1))

        # Return the maximum violation
        if len(cvs) == 1:
            return cvs[0]
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
