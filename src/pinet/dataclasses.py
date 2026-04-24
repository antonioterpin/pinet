"""This file contains dataclasses used to encapsulate inputs for the Pinet layer."""

import functools
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp

from .constants import Constants
from .constraints.non_linear_types import L2NormType, NonLinearConstraintType, SOCType

EQUILIBRATION_DEFAULT_MAX_ITER = Constants.EQUILIBRATION_DEFAULT_MAX_ITER
EQUILIBRATION_DEFAULT_TOL = Constants.EQUILIBRATION_DEFAULT_TOL
EQUILIBRATION_DEFAULT_ORD = Constants.EQUILIBRATION_DEFAULT_ORD
EQUILIBRATION_DEFAULT_COL_SCALING = Constants.EQUILIBRATION_DEFAULT_COL_SCALING
EQUILIBRATION_DEFAULT_UPDATE_MODE = Constants.EQUILIBRATION_DEFAULT_UPDATE_MODE
EQUILIBRATION_DEFAULT_SAFEGUARD = Constants.EQUILIBRATION_DEFAULT_SAFEGUARD

EXPECTED_BOUND_NDIM = 3
EXPECTED_PROJECTION_NDIM = 3
EXPECTED_SOC_TENSOR_NDIM = 3


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
class SocConstraintSpecification:
    """Dataclass representing inputs used in forming second-order cone constraints.

    Attributes:
        mask_u (Optional[jnp.ndarray]): Boolean mask indicating which variables
            are part of the cone constraint vector u. Shape (dimension,).
        mask_t (Optional[jnp.ndarray]): Boolean mask selecting a single
            scalar variable t that serves as the cone constraint parameter.
            Must have exactly one True value.
            Shape (dimension,).
        a (Optional[jnp.ndarray]): Coefficient matrix for the cone constraint vector u.
            Shape (batch_size, n_u_variables, 1) where n_u_variables is the number of True
            values in mask_u.
        b (Optional[jnp.ndarray]): Coefficient vector for the cone constraint parameter t.
            Shape (batch_size, 1, 1).
    """

    mask_u: jnp.ndarray | None = None
    mask_t: jnp.ndarray | None = None
    a: jnp.ndarray | None = None
    b: jnp.ndarray | None = None

    def update(self, **kwargs: object) -> "SocConstraintSpecification":
        """Update some attribute by keyword.

        Args:
            **kwargs: Field values to override.

        Returns:
            SocConstraintSpecification: Updated instance.
        """
        return replace(self, **kwargs)

    def validate(self) -> None:
        """Validate the soc constraint specification.

        NOTE: This checks cannot be done after tracing, but this function
        can be used to validate the inputs before tracing.
        """
        if getattr(self.mask_u, "dtype", None) != jnp.bool_:
            raise TypeError("mask_u must be a boolean array.")
        if getattr(self.mask_t, "dtype", None) != jnp.bool_:
            raise TypeError("mask_t must be a boolean array.")

        if getattr(self.mask_u, "ndim", None) != 1:
            raise ValueError(
                "mask_u must be a 1D array. "
                f"Received shape: {getattr(self.mask_u, 'shape', None)}."
            )
        if getattr(self.mask_t, "ndim", None) != 1:
            raise ValueError(
                "mask_t must be a 1D array. "
                f"Received shape: {getattr(self.mask_t, 'shape', None)}."
            )

        # Check that mask_u and mask_t have the same size
        if (
            self.mask_u is not None
            and self.mask_t is not None
            and hasattr(self.mask_u, "shape")
            and hasattr(self.mask_t, "shape")
        ):
            if self.mask_u.shape[0] != self.mask_t.shape[0]:
                raise ValueError(
                    "mask_u and mask_t must have the same size. "
                    f"Received mask_u shape: {self.mask_u.shape}, "
                    f"mask_t shape: {self.mask_t.shape}."
                )

        if (
            self.mask_t is not None
            and hasattr(self.mask_t, "sum")
            and self.mask_t.sum() != 1
        ):
            raise ValueError("mask_t must select exactly one element.")

        if (
            self.a is not None
            and hasattr(self.a, "ndim")
            and self.a.ndim != EXPECTED_SOC_TENSOR_NDIM
        ):
            raise ValueError(
                "a must have shape (batch_size, n_constraints, 1). "
                f"Received shape: {getattr(self.a, 'shape', None)}."
            )
        if (
            self.b is not None
            and hasattr(self.b, "ndim")
            and self.b.ndim != EXPECTED_SOC_TENSOR_NDIM
        ):
            raise ValueError(
                "b must have shape (batch_size, n_constraints, 1). "
                f"Received shape: {getattr(self.b, 'shape', None)}."
            )

        # Check that a and b are of the correct dimensions
        if self.a is not None and self.mask_u is not None:
            dim_a = self.a.shape[1]
            num_true_u = int(self.mask_u.sum())
            if dim_a != num_true_u:
                raise ValueError(
                    "The second dimension of a must match "
                    f"the number of True values in mask_u. "
                    f"Received a shape: {self.a.shape}, "
                    f"mask_u has {num_true_u} True values."
                )
        if self.b is not None:
            dim_b = self.b.shape[1]
            if dim_b != 1:
                raise ValueError(
                    "The second dimension of b must be 1. "
                    f"Received b shape: {self.b.shape}."
                )

    def to_nl_spec(self) -> "NonLinearSpecification":
        """Convert SocConstraintSpecification to NonLinearSpecification.

        Returns:
            NonLinearSpecification: A NonLinearSpecification instance with SOCType
                and the constraints from this specification.
        """
        if self.mask_t is None:
            raise ValueError("mask_t must be set to convert to NonLinearSpecification.")
        return NonLinearSpecification(
            nl_type=SOCType,
            A=jnp.empty((0, 0, self.mask_t.size)),
            a=self.a,
            f=jnp.empty((0, 1, self.mask_t.size)),
            b=self.b,
        )


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["A", "a", "f", "b"],
    meta_fields=["nl_type"],
)
@dataclass(frozen=True)
class NonLinearSpecification:
    """Dataclass representing inputs used in forming non-linear constraints.

    Attributes:
        nl_type (NonLinearConstraintType): The type of non-linear constraint
            (e.g., SOCType, L2NormType).
        A (jnp.ndarray): Matrix for the constraint. Shape (1, m, n)
            where m is the number of constraints and n is the number of variables.
        a (Optional[jnp.ndarray]): Coefficient array for the constraint.
            Shape (batch_size, m, 1) where m is the number of constraints.
        f (Optional[jnp.ndarray]): Optional RHS vector for the constraint.
            Shape (1, m, n) where m is the number of constraints
            and n is the number of variables.
        b (Optional[jnp.ndarray]): Optional scalar parameter for the constraint.
            Shape (batch_size, 1, 1).
    """

    nl_type: NonLinearConstraintType
    A: jnp.ndarray
    a: jnp.ndarray | None = None
    f: jnp.ndarray | None = None
    b: jnp.ndarray | None = None

    def update(self, **kwargs: object) -> "NonLinearSpecification":
        """Update some attribute by keyword.

        Args:
            **kwargs: Field values to override.

        Returns:
            NonLinearSpecification: Updated instance.
        """
        return replace(self, **kwargs)

    def _validate_type(self) -> None:
        if self.nl_type == L2NormType and self.f is not None:
            raise ValueError(
                "L2NormType with RHS (f) is not supported in NonLinearSpecification. "
                "Use SOCType instead."
            )
        if not isinstance(self.nl_type, NonLinearConstraintType):
            raise ValueError(
                f"nl_type must be a NonLinearConstraintType instance, "
                f"got {type(self.nl_type)}"
            )

    def _validate_batch_sizes(self) -> None:
        batch_sizes: list[int] = []
        for tensor in (self.A, self.a, self.f, self.b):
            if tensor is not None:
                batch_sizes.append(tensor.shape[0])
        if batch_sizes:
            non_one_sizes = [size for size in batch_sizes if size != 1]
            if len(set(non_one_sizes)) > 1:
                raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")
        if self.A.shape[0] != 1:
            raise ValueError(f"A must have batch size 1, got {self.A.shape[0]}")
        if self.f is not None and self.f.shape[0] != 1:
            raise ValueError(f"f must have batch size 1, got {self.f.shape[0]}")

    def _validate_dim_consistency(self) -> None:
        if self.a is not None and self.A.shape[1] != self.a.shape[1]:
            raise ValueError(
                f"A and a must have same constraint dimension: "
                f"{self.A.shape[1]} vs {self.a.shape[1]}"
            )
        if self.f is not None and self.A.shape[2] != self.f.shape[2]:
            raise ValueError(
                f"A and f must have same variable dimension: "
                f"{self.A.shape[2]} vs {self.f.shape[2]}"
            )
        if (
            self.f is not None
            and self.b is not None
            and self.f.shape[1] != self.b.shape[1]
        ):
            raise ValueError(
                f"f and b must have same constraint dimension: "
                f"{self.f.shape[1]} vs {self.b.shape[1]}"
            )
        if self.b is not None and self.b.shape[1] != 1:
            raise ValueError(
                f"b must be scalar (shape should be (batch_size, 1, 1)): "
                f"got {self.b.shape}"
            )

    def validate(self) -> None:
        """Validate the non-linear constraint specification."""
        self._validate_type()
        self._validate_batch_sizes()
        self._validate_dim_consistency()

    def to_primitive_spec(self) -> "SocConstraintSpecification":
        """Convert NonLinearSpecification to primitive constraint specification.

        Returns:
            SocConstraintSpecification: Equivalent primitive constraint spec.

        Raises:
            NotImplementedError: If the non-linear type is not supported.
        """
        if self.nl_type == SOCType:
            return SocConstraintSpecification(
                a=self.a,
                b=self.b,
            )
        raise NotImplementedError(
            f"Conversion to primitive spec not implemented for nl_type {self.nl_type}"
        )


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
        soc (Optional[SocConstraintSpecification]):
            Specification of the second-order cone constraints, if any.
        nl (Optional[list[NonLinearSpecification]]):
            Specification of the non-linear constraints, if any.
    """

    x: jnp.ndarray
    eq: EqualityConstraintsSpecification | None = None
    box: BoxConstraintSpecification | None = None
    soc: SocConstraintSpecification | None = None
    nl: list[NonLinearSpecification] | None = None

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

    max_iter: int = EQUILIBRATION_DEFAULT_MAX_ITER
    tol: float = EQUILIBRATION_DEFAULT_TOL
    ord: float = EQUILIBRATION_DEFAULT_ORD
    col_scaling: bool = EQUILIBRATION_DEFAULT_COL_SCALING
    update_mode: str = EQUILIBRATION_DEFAULT_UPDATE_MODE
    safeguard: bool = EQUILIBRATION_DEFAULT_SAFEGUARD

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
