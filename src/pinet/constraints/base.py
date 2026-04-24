"""Abstract class for constraint sets."""

from abc import abstractmethod

import equinox as eqx

from pinet._typing import BatchedScalar
from pinet.dataclasses import ProjectionInstance


class Constraint(eqx.Module):
    """Abstract class for constraint sets.

    Subclasses must implement ``project``, ``cv``, ``dim`` and ``n_constraints``.
    """

    @abstractmethod
    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project the input to the feasible region.

        Args:
            yraw: ProjectionInstance to project.

        Returns:
            The projected input.
        """

    @abstractmethod
    def cv(self, yraw: ProjectionInstance) -> BatchedScalar:
        """Compute the constraint violation.

        Args:
            yraw: ProjectionInstance to evaluate.

        Returns:
            The constraint violation for each point in the batch.
        """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimension of the constraint set."""

    @property
    @abstractmethod
    def n_constraints(self) -> int:
        """Return the number of constraints."""
