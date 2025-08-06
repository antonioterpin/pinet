"""Module for the Alternating Direction Method of Multipliers (ADMM) solver."""

from typing import Callable

import jax.numpy as jnp

from pinet.constraints import BoxConstraint, EqualityConstraint
from pinet.dataclasses import ProjectionInstance


def build_iteration_step(
    eq_constraint: EqualityConstraint,
    box_constraint: BoxConstraint,
    dim: int,
    scale: jnp.ndarray = 1.0,
) -> tuple[
    Callable[[ProjectionInstance, jnp.ndarray, float, float], ProjectionInstance],
    Callable[[ProjectionInstance], jnp.ndarray],
]:
    """Build the iteration and result retrieval step for the ADMM solver.

    See https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf for details.
    Args:
        eq_constraint (EqualityConstraint): (Lifted) Equality constraint.
        box_constraint (BoxConstraint): (Lifted) Box constraint.
        dim (int): Dimension of the original problem.
        scale (jnp.ndarray): Scaling of primal variables.

    Returns:
        tuple[
            Callable[[ProjectionInstance, jnp.ndarray, float, float], ProjectionInstance],
            Callable[[ProjectionInstance], ProjectionInstance]
        ]:
            The first element is the iteration step,
            the second element is the result retrieval step.
    """

    def iteration_step(
        sk: ProjectionInstance,
        yraw: jnp.ndarray,
        sigma: float = 1.0,
        omega: float = 1.7,
    ) -> ProjectionInstance:
        """One iteration of the ADMM solver.

        Args:
            sk (ProjectionInstance): State iterate for the ADMM solver.
                Shape (batch_size, lifted_dimension, 1).
            yraw (jnp.ndarray): Point to be projected. Shape (batch_size, dimension, 1).
            sigma (float, optional): ADMM parameter.
            omega (float, optional): ADMM parameter.

        Returns:
            jnp.ndarray: Next state iterate of the ADMM solver.
        """
        zk = eq_constraint.project(sk)
        # Reflection
        reflect = 2 * zk - sk.x
        tobox = jnp.concatenate(
            (
                (2 * sigma * scale * yraw + reflect[:, :dim, :])
                / (1 + 2 * sigma * scale**2),
                reflect[:, dim:, :],
            ),
            axis=1,
        )
        tk = box_constraint.project(sk.update(x=tobox))
        sk = sk.update(x=sk.x + omega * (tk - zk))
        return sk

    # The second element is used to extract the projection from the auxiliary
    return (iteration_step, lambda y: eq_constraint.project(y))
