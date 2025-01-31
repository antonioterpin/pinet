"""Module for the Alternating Direction Method of Multipliers (ADMM) solver."""

from typing import Callable, Tuple

import jax.numpy as jnp
from jax import jit

from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.box import BoxConstraint


def build_iteration_step(
    eq_constraint: EqualityConstraint,
    box_constraint: BoxConstraint,
    dim: int,
    sigma: float = 0.1,
    omega: float = 1.0,
) -> Tuple[
    Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    Callable[[jnp.ndarray], jnp.ndarray],
]:
    """Build the iteration and result retrieval step for the ADMM solver.

    See https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf for details.
    Args:
        eq_constraint (EqualityConstraint): (Lifted) Equality constraint.
        box_constraint (BoxConstraint): (Lifted) Box constraint.
        dim (int): Dimension of the original problem.
        sigma (float, optional): ADMM parameter.
        omega (float, optional): ADMM parameter.

    Returns:
        Tuple[
            Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
            Callable[[jnp.ndarray], jnp.ndarray]
        ]:
            The first element is the iteration step,
            the second element is the result retrieval step.
    """

    def iteration_step(xk: jnp.ndarray, xproj: jnp.ndarray):
        """One iteration of the ADMM solver.

        Args:
            xk (jnp.ndarray): State iterate for the ADMM solver.
            xproj (jnp.ndarray): Tensor to be projected.

        Returns:
            jnp.ndarray: Next state iterate of the ADMM solver.
        """
        dk = eq_constraint.project(xk)
        reflect = 2 * dk - xk
        tobox = jnp.concatenate(
            (
                (xproj + 1 / (2 * sigma) * reflect[:, :dim, :]) / (1 + 1 / (2 * sigma)),
                reflect[:, dim:, :],
            ),
            axis=1,
        )
        tk = box_constraint.project(tobox)
        xk = xk + omega * (tk - dk)
        return xk

    # The second element is used to extract the projection from the auxiliary
    return (jit(iteration_step), jit(lambda y: eq_constraint.project(y)[:, :dim, :]))
