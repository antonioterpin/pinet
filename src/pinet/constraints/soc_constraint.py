"""Second order cone constraint module."""

import equinox as eqx
import jax
import jax.numpy as jnp

from pinet._typing import BatchedIneqBound, BatchedScalar, Mask1D
from pinet.constants import Constants
from pinet.constraints.base import Constraint
from pinet.dataclasses import ProjectionInstance, SocConstraintSpecification

SOC_CONSTRAINT_EPSILON = Constants.SOC_CONSTRAINT_EPSILON


class SocConstraint(Constraint):
    """Second-order cone (SOC) constraint set.

    The SOC constraint set is defined as:
    || y_[mask_u] + a ||_2 <= y_[mask_t] + b
    where mask_t has one True element and b is a scalar.

    Attributes:
        mask_u: Mask selecting the variables that form the cone vector ``u``.
        mask_t: Mask selecting the scalar variable ``t``.
        a: Offset vector added to ``u``.
        b: Scalar offset added to ``t``.
        eps: Small epsilon for numerical stability of the projection direction.
    """

    mask_u: Mask1D
    mask_t: Mask1D
    a: BatchedIneqBound
    b: BatchedScalar
    eps: float = eqx.field(static=True)
    _dim: int = eqx.field(static=True)

    def __init__(
        self,
        socspec: SocConstraintSpecification,
    ) -> None:
        """Initialize the SOC constraint.

        Args:
            socspec: Specification of the SOC constraint.
        """
        if socspec.mask_u is None or socspec.mask_t is None:
            raise ValueError("Both mask_u and mask_t must be provided.")
        socspec.validate()
        self.mask_u = socspec.mask_u
        self.mask_t = socspec.mask_t
        self._dim = int(self.mask_u.shape[0])
        # For numerical stability of projection
        self.eps = SOC_CONSTRAINT_EPSILON
        self.a = (
            socspec.a
            if socspec.a is not None
            else jnp.zeros((1, int(self.mask_u.sum()), 1))
        )
        self.b = socspec.b if socspec.b is not None else jnp.zeros((1, 1, 1))

    def unpack_instance(
        self, inp: ProjectionInstance
    ) -> tuple[
        Mask1D,
        BatchedIneqBound,
        Mask1D,
        BatchedScalar,
        BatchedIneqBound,
        BatchedScalar,
    ]:
        """Unpack point to project in convenient form.

        Args:
            inp: ProjectionInstance to unpack.

        Returns:
            Tuple of (mask_u, u, mask_t, t, a, b) used by projection/cv.
        """
        if inp.soc and (inp.soc.mask_u is not None or inp.soc.mask_t is not None):
            raise ValueError(
                "Per-instance masks for SOC constraints are not supported. "
                "Only provide point-to-project."
            )
        a = inp.soc.a if inp.soc and (inp.soc.a is not None) else self.a
        b = inp.soc.b if inp.soc and (inp.soc.b is not None) else self.b
        mask_u = self.mask_u
        u = inp.x[:, mask_u, :] + a
        mask_t = self.mask_t
        t = inp.x[:, mask_t, :] + b
        return mask_u, u, mask_t, t, a, b

    def project(self, y_raw: ProjectionInstance) -> ProjectionInstance:
        """Project onto SOC constraints.

        Args:
            y_raw: ProjectionInstance to project.
                The .x attribute is the point to project.

        Returns:
            The projected point for each point in the batch.
        """
        mask_u, u, mask_t, t, a, b = self.unpack_instance(y_raw)
        norm_u = jnp.linalg.norm(u, axis=1, keepdims=True)
        z: jax.Array = jnp.concatenate([u, t], axis=1)

        proj1: jax.Array = z
        proj2: jax.Array = jnp.zeros_like(z)
        direction: jax.Array = jnp.concatenate(
            [u / (norm_u + self.eps), jnp.ones_like(t)], axis=1
        )
        proj3: jax.Array = (t + norm_u) / 2 * direction

        when1 = norm_u <= t
        when2 = norm_u <= -t
        inner: jax.Array = jnp.where(when2, proj2, proj3)
        final_proj: jax.Array = jnp.where(when1, proj1, inner)

        # Coerce to jax.Array before using ``.at`` (jax-only).
        x = jnp.asarray(y_raw.x)
        return y_raw.update(
            x=x.at[:, mask_u, :]
            .set(final_proj[:, :-1, :] - a)
            .at[:, mask_t, :]
            .set(final_proj[:, -1:, :] - b)
        )

    def cv(self, y_raw: ProjectionInstance) -> BatchedScalar:
        """Compute the constraint violation.

        The SOC constraint is: ||u + a||_2 <= t + b. The violation is
        ``max(0, ||u + a||_2 - (t + b))``.

        Args:
            y_raw: ProjectionInstance to evaluate.

        Returns:
            The constraint violation for each point in the batch.
        """
        _mask_u, u, _mask_t, t, _a, _b = self.unpack_instance(y_raw)
        norm_u = jnp.linalg.norm(u, axis=1, keepdims=True)

        # Constraint violation: ||u||_2 - t (where u and t already include a and b)
        violation = norm_u - t

        return jnp.maximum(violation, 0)

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set."""
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return 1
