"""Abstract class for constraint sets."""

import equinox as eqx

from pinet._typing import BatchedScalar
from pinet.dataclasses import ProjectionInstance


class Constraint(eqx.Module):
    """Abstract class for constraint sets.

    Subclasses must override ``project``, ``cv``, ``dim`` and ``n_constraints``.
    The base implementations raise ``NotImplementedError`` — ``@abstractmethod``
    alone is not runtime-enforced on ``eqx.Module`` because it does not use
    ``ABCMeta`` as its metaclass.
    """

    def project(self, y_raw: ProjectionInstance) -> ProjectionInstance:
        """Project the input to the feasible region.

        Args:
            y_raw: ProjectionInstance to project.

        Returns:
            The projected input.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.project must be implemented by a subclass."
        )

    def cv(self, y_raw: ProjectionInstance) -> BatchedScalar:
        """Compute the constraint violation.

        Args:
            y_raw: ProjectionInstance to evaluate.

        Returns:
            The constraint violation for each point in the batch.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.cv must be implemented by a subclass."
        )

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.dim must be implemented by a subclass."
        )

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints.

        Raises:
            NotImplementedError: Always; subclasses must override.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.n_constraints must be implemented by a subclass."
        )
