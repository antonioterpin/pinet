"""This file contains dataclasses used to encapsulate inputs for the Pinet layer."""

from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp

from ._typing import (
    BatchedEqMatrix,
    BatchedEqPinv,
    BatchedIneqBound,
    BatchedPrimal,
    BatchedRHS,
    BatchedScalar,
    Mask1D,
    NLLinearRHS,
    NLMatrix,
)
from .constants import Constants
from .constraints.non_linear_types import L2NormType, NonLinearConstraintType, SOCType

EQUILIBRATION_DEFAULT_MAX_ITER = Constants.EQUILIBRATION_DEFAULT_MAX_ITER
EQUILIBRATION_DEFAULT_TOL = Constants.EQUILIBRATION_DEFAULT_TOL
EQUILIBRATION_DEFAULT_ORD = Constants.EQUILIBRATION_DEFAULT_ORD
EQUILIBRATION_DEFAULT_COL_SCALING = Constants.EQUILIBRATION_DEFAULT_COL_SCALING
EQUILIBRATION_DEFAULT_UPDATE_MODE = Constants.EQUILIBRATION_DEFAULT_UPDATE_MODE
EQUILIBRATION_DEFAULT_SAFEGUARD = Constants.EQUILIBRATION_DEFAULT_SAFEGUARD


class EqualityConstraintsSpecification(eqx.Module):
    """Dataclass representing inputs used in forming equality constraints.

    Attributes:
        b: Vector representing the RHS of the equality constraint.
        a_mat: Matrix representing the LHS of the equality constraint.
        a_mat_pinv: The pseudoinverse of ``a_mat``.
    """

    b: BatchedRHS | None = None
    a_mat: BatchedEqMatrix | None = None
    a_mat_pinv: BatchedEqPinv | None = None

    def validate(self) -> None:
        """Validate the equality constraints specification.

        Must run before tracing; the checks are not jit-traceable.

        Raises:
            ValueError: If a_mat is provided but b is not.
        """
        if self.a_mat is not None and self.b is None:
            raise ValueError("If a_mat is provided, b must also be provided.")

    def update(self, **kwargs: object) -> "EqualityConstraintsSpecification":
        """Return a copy with the given fields overridden.

        Args:
            **kwargs: New values for fields to override.

        Returns:
            Updated instance.
        """
        return replace(self, **kwargs)


class BoxConstraintSpecification(eqx.Module):
    """Dataclass representing inputs used in forming box constraints.

    Attributes:
        lb: Lower bound of the box.
        ub: Upper bound of the box.
        mask: Mask to apply the constraint only to some dimensions.
    """

    lb: BatchedRHS | None = None
    ub: BatchedRHS | None = None
    mask: Mask1D | None = None

    def update(self, **kwargs: object) -> "BoxConstraintSpecification":
        """Return a copy with the given fields overridden.

        Args:
            **kwargs: New values for fields to override.

        Returns:
            Updated instance.
        """
        return replace(self, **kwargs)

    def _validate_bound_compatibility(self) -> None:
        if self.lb is None or self.ub is None:
            return
        if not jnp.all(self.lb <= self.ub):
            raise ValueError("Lower bound must be less than or equal to the upper bound.")

    def _validate_mask(self) -> None:
        if self.mask is None:
            return
        dim = getattr(self.lb, "shape", None) or getattr(self.ub, "shape", None)
        if dim is not None and dim[1] != int(jnp.sum(self.mask)):
            raise ValueError(
                "Number of active entries in the mask must match the bounds. "
                f"Received mask shape: {getattr(self.mask, 'shape', None)}, "
                f"bound shape: {dim}."
            )

    def validate(self) -> None:
        """Validate the box constraint specification.

        The ranks and dtypes of ``lb``/``ub``/``mask`` (and that bounds share
        a shape up to broadcasting) are enforced by the type system at
        construction time via the ``PINET_RUNTIME_CHECK`` beartype hook; only
        the non-type-level invariants are checked here. Must run before
        tracing; the checks are not jit-traceable.

        Raises:
            ValueError: If neither bound is provided, the lower bound exceeds
                the upper bound, or the mask's active count disagrees with the
                bounds.
        """
        if self.lb is None and self.ub is None:
            raise ValueError("At least one of lower or upper bounds must be provided.")

        self._validate_bound_compatibility()
        self._validate_mask()


class SocConstraintSpecification(eqx.Module):
    """Dataclass representing inputs used in forming second-order cone constraints.

    Attributes:
        mask_u: Boolean mask selecting the variables that form the cone vector ``u``.
        mask_t: Boolean mask selecting the scalar variable ``t``; must have exactly
            one ``True`` entry.
        a: Offset vector added to ``u``.
        b: Scalar offset added to ``t``.
    """

    mask_u: Mask1D | None = None
    mask_t: Mask1D | None = None
    a: BatchedIneqBound | None = None
    b: BatchedScalar | None = None

    def update(self, **kwargs: object) -> "SocConstraintSpecification":
        """Return a copy with the given fields overridden.

        Args:
            **kwargs: New values for fields to override.

        Returns:
            Updated instance.
        """
        return replace(self, **kwargs)

    def validate(self) -> None:
        """Validate the SOC constraint specification.

        The dtypes and ranks of the masks/offsets, that ``mask_u`` and
        ``mask_t`` share a size, and that ``b`` is a scalar are enforced by the
        type system at construction time via the ``PINET_RUNTIME_CHECK``
        beartype hook; only the value-level invariants (which depend on the
        runtime contents of the masks) are checked here. Must run before
        tracing; the checks are not jit-traceable.

        Raises:
            ValueError: If ``mask_t`` does not select exactly one element, or
                the second dimension of ``a`` does not match the number of
                ``True`` values in ``mask_u``.
        """
        if (
            self.mask_t is not None
            and hasattr(self.mask_t, "sum")
            and self.mask_t.sum() != 1
        ):
            raise ValueError("mask_t must select exactly one element.")

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

    def to_nl_spec(self) -> "NonLinearSpecification":
        """Convert SocConstraintSpecification to NonLinearSpecification.

        Returns:
            A ``NonLinearSpecification`` with ``SOCType`` and the constraints from
            this specification.
        """
        self.validate()
        assert self.mask_t is not None  # narrowed by validate()
        # ``a_mat`` and ``f`` are placeholders for SOC (which uses masks instead
        # of the full a_mat/f machinery). Use batch size 1 to satisfy the
        # documented ``NLMatrix`` / ``NLLinearRHS`` shape contracts
        # (``"1 m n"`` and ``"1 1 n"``); the constraint dimensions
        # ``m=0``/``1`` and the variable dimension ``n=mask_t.size`` keep the
        # placeholders empty.
        return NonLinearSpecification(
            nl_type=SOCType,
            a_mat=jnp.empty((1, 0, self.mask_t.size)),
            a=self.a,
            f=jnp.empty((1, 1, self.mask_t.size)),
            b=self.b,
        )


class NonLinearSpecification(eqx.Module):
    """Dataclass representing inputs used in forming non-linear constraints.

    Attributes:
        nl_type: The type of non-linear constraint (e.g., ``SOCType``, ``L2NormType``).
        a_mat: Matrix for the constraint (single shared matrix across the batch).
        a: Per-instance coefficient vector for the constraint.
        f: Optional RHS vector for the constraint (single shared vector).
        b: Optional per-instance scalar parameter for the constraint.
    """

    nl_type: NonLinearConstraintType = eqx.field(static=True)
    a_mat: NLMatrix
    a: BatchedRHS | None = None
    f: NLLinearRHS | None = None
    b: BatchedScalar | None = None

    def update(self, **kwargs: object) -> "NonLinearSpecification":
        """Update some attribute by keyword.

        Args:
            **kwargs: Field values to override.

        Returns:
            Updated instance.
        """
        return replace(self, **kwargs)

    def _validate_type(self) -> None:
        if self.nl_type == L2NormType and self.f is not None:
            raise ValueError(
                "L2NormType with RHS (f) is not supported in NonLinearSpecification. "
                "Use SOCType instead."
            )
        # Single owner of the "which nl_type the library supports" invariant.
        # The parser and to_primitive_spec rely on this: any spec that passes
        # validate() is guaranteed to carry a supported type, so they need no
        # parallel guard. A future NonLinearConstraintType member is rejected
        # here at construction instead of being silently mis-projected.
        if self.nl_type not in (SOCType, L2NormType):
            raise ValueError(
                f"nl_type must be SOCType or L2NormType, got {self.nl_type!r}."
            )

    def validate(self) -> None:
        """Validate the non-linear constraint specification.

        Batch sizes and the cross-tensor shape relations between ``a_mat``,
        ``a``, ``f`` and ``b`` are enforced by the type system at construction
        time via the ``PINET_RUNTIME_CHECK`` beartype hook; only the supported
        ``nl_type`` invariant (which is not expressible in the type) is checked
        here.
        """
        self._validate_type()

    def to_primitive_spec(self) -> "SocConstraintSpecification":
        """Convert NonLinearSpecification to primitive constraint specification.

        Returns:
            Equivalent primitive constraint spec.

        Raises:
            ValueError: If ``nl_type`` is not a supported type.
        """
        self._validate_type()
        # SOCType (with or without ``f``) and L2NormType both project via
        # the SOC primitive: the parser already lifts ``A x + a`` and
        # ``f x + b`` (or ``b`` alone for L2) into auxiliary variables —
        # here A is ``a_mat``, a is ``a``, f is ``f``, b is ``b`` — so the
        # downstream SOC sees the original ``a`` / ``b`` offsets.
        return SocConstraintSpecification(a=self.a, b=self.b)


class ProjectionInstance(eqx.Module):
    """A dataclass for encapsulating model input parameters.

    Attributes:
        x: The point to be projected.
        eq: Specification of the equality constraints, if any.
        box: Specification of the box constraints, if any.
        soc: Specification of the second-order cone constraints, if any.
        nl: Specification of the non-linear constraints, if any.
    """

    x: BatchedPrimal
    eq: EqualityConstraintsSpecification | None = None
    box: BoxConstraintSpecification | None = None
    soc: SocConstraintSpecification | None = None
    nl: list[NonLinearSpecification] | None = None

    def validate(self) -> None:
        """Validate the projection instance.

        The rank of ``x`` (shape ``(batch_size, dimension, 1)``) is enforced by
        the type system at construction time via the ``PINET_RUNTIME_CHECK``
        beartype hook, so there is nothing left to validate here that is not
        jit-traceable. Kept for API symmetry with the constraint specs and as
        the documented pre-tracing entry point.
        """

    def update(self, **kwargs: object) -> "ProjectionInstance":
        """Return a copy with the given fields overridden.

        Args:
            **kwargs: New values for fields to override.

        Returns:
            Updated instance.
        """
        return replace(self, **kwargs)


class EquilibrationParams(eqx.Module):
    """A dataclass for encapsulating the equilibration parameters.

    Attributes:
        max_iter: Maximum number of iterations for the equilibration.
        tol: Tolerance for convergence of the equilibration.
        ord: Order of the norm used for convergence check.
        col_scaling: Whether to apply column scaling.
        update_mode: Update mode for the equilibration. Available options are:

            * ``"Jacobi"``: compute both row and column norms and update.
            * ``"Gauss"``: compute row, update, compute column, update.

        safeguard: Check if the condition number of ``a_mat`` has decreased.
    """

    max_iter: int = eqx.field(static=True, default=EQUILIBRATION_DEFAULT_MAX_ITER)
    tol: float = eqx.field(static=True, default=EQUILIBRATION_DEFAULT_TOL)
    ord: float = eqx.field(static=True, default=EQUILIBRATION_DEFAULT_ORD)
    col_scaling: bool = eqx.field(static=True, default=EQUILIBRATION_DEFAULT_COL_SCALING)
    update_mode: str = eqx.field(static=True, default=EQUILIBRATION_DEFAULT_UPDATE_MODE)
    safeguard: bool = eqx.field(static=True, default=EQUILIBRATION_DEFAULT_SAFEGUARD)

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
            Updated instance.
        """
        return replace(self, **kwargs)
