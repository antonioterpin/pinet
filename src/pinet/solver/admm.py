"""Module for the Alternating Direction Method of Multipliers (ADMM) solver."""

from typing import Callable

import jax.numpy as jnp

from pinet.constraints import BoxConstraint, EqualityConstraint
from pinet.dataclasses import Inputs


def build_iteration_step(
    eq_constraint: EqualityConstraint,
    box_constraint: BoxConstraint,
    dim: int,
    scale: jnp.ndarray = 1.0,
) -> tuple[
    Callable[[Inputs, jnp.ndarray, float, float], Inputs],
    Callable[[Inputs], jnp.ndarray],
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
            Callable[[Inputs, jnp.ndarray, float, float], Inputs],
            Callable[[Inputs], Inputs]
        ]:
            The first element is the iteration step,
            the second element is the result retrieval step.
    """

    def iteration_step(
        xk: Inputs,
        xproj: jnp.ndarray,
        sigma: float = 1.0,
        omega: float = 1.7,
    ) -> Inputs:
        """One iteration of the ADMM solver.

        Args:
            xk (jnp.ndarray): State iterate for the ADMM solver.
                Shape (batch_size, lifted_dimension, 1).
            xproj (jnp.ndarray): Point to be projected.
                Shape (batch_size, dimension, 1).
            sigma (float, optional): ADMM parameter.
            omega (float, optional): ADMM parameter.

        Returns:
            jnp.ndarray: Next state iterate of the ADMM solver.
        """
        dk = eq_constraint.project(xk)
        # Reflection
        reflect = 2 * dk - xk.x
        tobox = jnp.concatenate(
            (
                (2 * sigma * scale * xproj + reflect[:, :dim, :])
                / (1 + 2 * sigma * scale**2),
                reflect[:, dim:, :],
            ),
            axis=1,
        )
        tk = box_constraint.project(xk.update(x=tobox))
        xk = xk.update(x=xk.x + omega * (tk - dk))
        return xk

    # The second element is used to extract the projection from the auxiliary
    return (iteration_step, lambda y: eq_constraint.project(y))
