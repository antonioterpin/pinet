"""This file contains dataclasses used to encapsulate inputs for the Pinet layer."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

EXPECTED_BOUND_NDIM = 3
EXPECTED_PROJECTION_NDIM = 3


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EqualityConstraintsSpecification:
    """Dataclass representing inputs used in forming equality constraints.

    Attributes:
        b (Optional[jnp.ndarray]): Vector representing the RHS of the equality constraint.
            Shape (batch_size, n_constraints, 1)
        a_dyn (Optional[jnp.ndarray]): Matrix representing the LHS of the equality
            constraint.
            Shape (batch_size, n_constraints, dimension).
        a_dyn_pinv (Optional[jnp.ndarray]): The pseudoinverse of the matrix a_dyn.
            Shape (batch_size, dimension, n_constraints).
    """

    b: jnp.ndarray | None = None
    a_dyn: jnp.ndarray | None = None
    a_dyn_pinv: jnp.ndarray | None = None

    def validate(self) -> None:
        """Validate the equality constraints specification.

        NOTE: This checks cannot be done after tracing, but this function
        can be used to validate the inputs before tracing.

        Raises:
            ValueError: If a_dyn is provided but b is not.
        """
        if self.a_dyn is not None and self.b is None:
            raise ValueError("If a_dyn is provided, b must also be provided.")

    def update(self, **kwargs: object) -> "EqualityConstraintsSpecification":
        """Update some attribute by keyword.

        Args:
            **kwargs: Keyword arguments matching field names to update.

        Returns:
            EqualityConstraintsSpecification: Updated instance.
        """
        return replace(self, **kwargs)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BoxConstraintSpecification:
    """Dataclass representing inputs used in forming box constraints.

    Attributes:
        lb (jnp.ndarray): Lower bound of the box. Shape (batch_size, n_constraints, 1).
        ub (jnp.ndarray): Upper bound of the box. Shape (batch_size, n_constraints, 1).
        mask (jnp.ndarray | None):
            Mask to apply the constraint only to some dimensions.
    """

    lb: jnp.ndarray | None = None
    ub: jnp.ndarray | None = None
    mask: jnp.ndarray | None = None

    def update(self, **kwargs: object) -> "BoxConstraintSpecification":
        """Update some attribute by keyword.

        Args:
            **kwargs: Keyword arguments matching field names to update.

        Returns:
            BoxConstraintSpecification: Updated instance.
        """
        return replace(self, **kwargs)

    def _validate_bound_shapes(self) -> None:
        if (
            self.lb is not None
            and hasattr(self.lb, "ndim")
            and self.lb.ndim != EXPECTED_BOUND_NDIM
        ):
            raise ValueError(
                "Lower bound must have shape (batch_size, n_constraints, 1). "
                f"Received shape: {getattr(self.lb, 'shape', None)}."
            )
        if (
            self.ub is not None
            and hasattr(self.ub, "ndim")
            and self.ub.ndim != EXPECTED_BOUND_NDIM
        ):
            raise ValueError(
                "Upper bound must have shape (batch_size, n_constraints, 1). "
                f"Received shape: {getattr(self.ub, 'shape', None)}."
            )

    def _validate_bound_compatibility(self) -> None:
        if self.lb is None or self.ub is None:
            return
        if hasattr(self.lb, "shape") and hasattr(self.ub, "shape"):
            if self.lb.shape[1:] != self.ub.shape[1:]:
                raise ValueError(
                    "Lower and upper bounds must have the same shape. "
                    f"Received shapes: {self.lb.shape} and {self.ub.shape}."
                )
            if (
                self.lb.shape[0] != self.ub.shape[0]
                and self.lb.shape[0] != 1
                and self.ub.shape[0] != 1
            ):
                raise ValueError(
                    "Batch size of lower and upper bounds must be the same "
                    "or one of them must be 1. "
                    f"Received shapes: {self.lb.shape} and {self.ub.shape}."
                )

        if not jnp.all(self.lb <= self.ub):
            raise ValueError("Lower bound must be less than or equal to the upper bound.")

    def _validate_mask(self) -> None:
        if self.mask is None:
            return
        if getattr(self.mask, "dtype", None) != jnp.bool_:
            raise TypeError("Mask must be a boolean array.")
        if getattr(self.mask, "ndim", None) != 1:
            raise ValueError("Mask must be a 1D array.")

        dim = getattr(self.lb, "shape", None) or getattr(self.ub, "shape", None)
        if dim is not None and dim[1] != int(jnp.sum(self.mask)):
            raise ValueError(
                "Number of active entries in the mask must match the bounds. "
                f"Received mask shape: {getattr(self.mask, 'shape', None)}, "
                f"bound shape: {dim}."
            )

    def validate(self) -> None:
        """Validate the box constraint specification.

        NOTE: This checks cannot be done after tracing, but this function
        can be used to validate the inputs before tracing.

        Raises:
            ValueError: If bounds are invalid or inconsistent.
        """
        if self.lb is None and self.ub is None:
            raise ValueError("At least one of lower or upper bounds must be provided.")

        self._validate_bound_shapes()
        self._validate_bound_compatibility()
        self._validate_mask()


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ProjectionInstance:
    """A dataclass for encapsulating model input parameters.

    Attributes:
        x (jnp.ndarray): The point to be projected.
            Shape (batch_size, dimension, 1)
        eq (Optional[EqualityConstraintsSpecification]):
            Specification of the equality constraints, if any.
        box (Optional[BoxConstraintSpecification]):
            Specification of the box constraints, if any.
    """

    x: jnp.ndarray
    eq: EqualityConstraintsSpecification | None = None
    box: BoxConstraintSpecification | None = None

    def validate(self) -> None:
        """Validate the projection instance.

        NOTE: This checks cannot be done after tracing, but this function
        can be used to validate the inputs before tracing.

        Raises:
            ValueError: If x does not have shape (batch_size, dimension, 1).
        """
        if self.x.ndim != EXPECTED_PROJECTION_NDIM:
            raise ValueError(
                "x must have shape (batch_size, dimension, 1). "
                f"Received shape: {self.x.shape}."
            )

    def update(self, **kwargs: object) -> "ProjectionInstance":
        """Update some attribute by keyword.

        Args:
            **kwargs: Keyword arguments matching field names to update.

        Returns:
            ProjectionInstance: Updated instance.
        """
        return replace(self, **kwargs)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EquilibrationParams:
    """A dataclass for encapsulating the equilibration parameters.

    Attributes:
        max_iter (int): Maximum number of iterations for the equilibration.
        tol (float): Tolerance for convergence of the equilibration.
        ord (float): Order of the norm used for convergence check.
        col_scaling (bool): Whether to apply column scaling.
        update_mode (str): Update mode for the equilibration.
            Available options are:
                - "Jacobi" means compute both row and column norms and update.
                - "Gauss" means compute row, update, compute column, update.
        safeguard (bool): Check if the condition number of a_dyn has decreased.
    """

    max_iter: int = 0
    tol: float = 1e-3
    ord: float = 2.0
    col_scaling: bool = False
    update_mode: str = "Gauss"
    safeguard: bool = False

    def validate(self) -> None:
        """Validate the equilibration parameters.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative.")
        if self.tol <= 0:
            raise ValueError("tol must be positive.")
        if self.ord not in [1, 2, float("inf")]:
            raise ValueError("ord must be 1, 2, or infinity.")
        if self.update_mode not in ["Gauss", "Jacobi"]:
            raise ValueError('update_mode must be either "Gauss" or "Jacobi".')

    def update(self, **kwargs: object) -> "EquilibrationParams":
        """Update some attribute by keyword.

        Args:
            **kwargs: Keyword arguments matching field names to update.

        Returns:
            EquilibrationParams: Updated instance.
        """
        return replace(self, **kwargs)
