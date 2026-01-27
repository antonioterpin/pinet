"""Second order cone constraint module."""

import jax.numpy as jnp

from pinet.constraints.base import Constraint
from pinet.dataclasses import ProjectionInstance, SocConstraintSpecification


class SocConstraint(Constraint):
    """Second-order cone (SOC) constraint set.

    The SOC constraint set is defined as:
    || y_[0:dim-2] + a ||_2 <= y_[dim-1] + b
    """

    def __init__(
        self,
        socspec: SocConstraintSpecification,
    ) -> None:
        """Initialize the SOC constraint.

        Args:
            socspec (SocConstraintSpecification): Specification of the box constraint.
        """
        socspec.validate()
        self.a = socspec.a
        self.b = socspec.b
        if socspec.mask_u is None or socspec.mask_t is None:
            raise ValueError("Both mask_u and mask_t must be provided.")
        self._dim = socspec.mask_u.shape[0]
        # For numerical stability of projection
        self.eps = 1e-12
        self.mask_u = socspec.mask_u
        self.mask_t = socspec.mask_t
        if self.a is None:
            self.a = jnp.zeros((1, self.mask_u.sum(), 1))
        if self.b is None:
            self.b = jnp.zeros((1, 1, 1))

    def unpack_instance(
        self, inp: ProjectionInstance
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Unpack point to project in convenient form.

        Args:
            inp (ProjectionInstance): ProjectionInstance to unpack.
        """
        if inp.soc and (inp.soc.mask_u is not None or inp.soc.mask_t is not None):
            raise ValueError(
                "Per-instance masks for SOC constraints are not supported."
                "Only provide point-to-project."
            )
        a = inp.soc.a if inp.soc and (inp.soc.a is not None) else self.a
        b = inp.soc.b if inp.soc and (inp.soc.b is not None) else self.b
        mask_u = self.mask_u
        u = inp.x[:, mask_u, :] + a
        mask_t = self.mask_t
        t = inp.x[:, mask_t, :] + b
        return mask_u, u, mask_t, t, a, b

    def project(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project onto SOC constraints.

        Args:
            yraw (ProjectionInstance): ProjectionInstance to projection.
                The .x attribute is the point to project.

        Returns:
            ProjectionInstance: The projected point for each point in the batch.
        """
        mask_u, u, mask_t, t, a, b = self.unpack_instance(yraw)
        norm_u = jnp.linalg.norm(u, axis=1, keepdims=True)
        z = jnp.concatenate((u, t), axis=1)

        proj1 = z
        proj2 = jnp.zeros_like(z)
        proj3 = (
            (t + norm_u)
            / 2
            * jnp.concatenate((u / (norm_u + self.eps), jnp.ones_like(t)), axis=1)
        )

        when1 = norm_u <= t
        when2 = norm_u <= -t
        final_proj = jnp.where(when1, proj1, jnp.where(when2, proj2, proj3))

        return yraw.update(
            x=yraw.x.at[:, mask_u, :]
            .set(final_proj[:, :-1, :] - a)
            .at[:, mask_t, :]
            .set(final_proj[:, -1:, :] - b)
        )

    def cv(self, yraw: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        The SOC constraint is: ||u + a||_2 <= t + b
        The violation is: max(0, ||u + a||_2 - (t + b))

        Args:
            yraw (ProjectionInstance): ProjectionInstance to evaluate.

        Returns:
            jnp.ndarray: The constraint violation for each point in the batch.
                Shape (batch_size, 1, 1).
        """
        mask_u, u, mask_t, t, a, b = self.unpack_instance(yraw)
        norm_u = jnp.linalg.norm(u, axis=1, keepdims=True)

        # Constraint violation: ||u||_2 - t (where u and t already include a and b)
        violation = norm_u - t

        return jnp.maximum(violation, 0)

    @property
    def dim(self) -> int:
        """Return the dimension of the constraint set.

        Returns:
            int: The dimension of the constraint set.
        """
        return self._dim

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints.

        Returns:
            int: The number of constraints.
        """
        return 1
