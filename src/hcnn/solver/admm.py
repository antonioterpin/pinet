"""Module for the Alternating Direction Method of Multipliers (ADMM) solver."""

from typing import Callable, Tuple

import jax.numpy as jnp

from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.box import BoxConstraint

# TODO: Adapt the final step of all versions.


def build_iteration_step(
    eq_constraint: EqualityConstraint,
    box_constraint: BoxConstraint,
    dim: int,
    sigma: float = 1.0,
    omega: float = 1.7,
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
    return (iteration_step, lambda y: eq_constraint.project(y))


def build_iteration_step_vb(
    eq_constraint: EqualityConstraint,
    box_constraint: BoxConstraint,
    dim: int,
    scale: jnp.ndarray,
    sigma: float = 1.0,
    omega: float = 1.7,
) -> Tuple[
    Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    Callable[[jnp.ndarray], jnp.ndarray],
]:
    """Build the iteration and result retrieval step for the ADMM solver.

    Assumes variable b vector for equality.
    See https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf for details.
    Args:
        eq_constraint (EqualityConstraint): (Lifted) Equality constraint.
        box_constraint (BoxConstraint): (Lifted) Box constraint.
        b: Right-hand side vector for equality.
        dim (int): Dimension of the original problem.
        scale (jnp.ndarray): Scaling of primal variables.
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

    def iteration_step(xk: jnp.ndarray, xproj: jnp.ndarray, b: jnp.ndarray):
        """One iteration of the ADMM solver.

        Args:
            xk (jnp.ndarray): State iterate for the ADMM solver.
            xproj (jnp.ndarray): Tensor to be projected.

        Returns:
            jnp.ndarray: Next state iterate of the ADMM solver.
        """
        dk = eq_constraint.project(xk, b)
        reflect = 2 * dk - xk
        tobox = jnp.concatenate(
            (
                (scale * xproj + 1 / (2 * sigma) * reflect[:, :dim, :])
                / (scale**2 + 1 / (2 * sigma)),
                reflect[:, dim:, :],
            ),
            axis=1,
        )
        tk = box_constraint.project(tobox)
        xk = xk + omega * (tk - dk)
        return xk

    # The second element is used to extract the projection from the auxiliary
    return (iteration_step, lambda y, b: eq_constraint.project(y, b))


def build_iteration_step_vAb(
    eq_constraint: EqualityConstraint,
    box_constraint: BoxConstraint,
    dim: int,
    sigma: float = 1.0,
    omega: float = 1.7,
) -> Tuple[
    Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    Callable[[jnp.ndarray], jnp.ndarray],
]:
    """Build the iteration and result retrieval step for the ADMM solver.

    Assumes variable b vector for equality.
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

    def iteration_step(
        xk: jnp.ndarray,
        xproj: jnp.ndarray,
        b: jnp.ndarray,
        A: jnp.ndarray,
        Apinv: jnp.ndarray,
    ):
        """One iteration of the ADMM solver.

        Args:
            xk (jnp.ndarray): State iterate for the ADMM solver.
            xproj (jnp.ndarray): Tensor to be projected.
            b: Right-hand side vector for equality.
            A: Left-hand side matrix for equality.
            Apinv: Pseudo-inverse of A.

        Returns:
            jnp.ndarray: Next state iterate of the ADMM solver.
        """
        dk = eq_constraint.project(xk, b, A, Apinv)
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
    return (
        iteration_step,
        lambda y, b, A, Apinv: eq_constraint.project(y, b, A, Apinv)[:, :dim, :],
    )
