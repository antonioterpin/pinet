"""Module for the Alternating Direction Method of Multipliers (ADMM) solver."""

from collections.abc import Callable

import jax.numpy as jnp

from pinet.constraints import (
    AffineInequalityConstraint,
    BoxConstraint,
    ConstraintParser,
    EqualityConstraint,
)
from pinet.dataclasses import ProjectionInstance


def initialize(
    yraw: ProjectionInstance,
    ineq_constraint: AffineInequalityConstraint | None,
    box_constraint: BoxConstraint | None,
    dim: int,
    dim_lifted: int,
    d_r: jnp.ndarray,
) -> ProjectionInstance:
    """Initialize the ADMM solver state.

    Args:
        yraw: Point to be projected. Shape (batch_size, dimension, 1).
        ineq_constraint: Inequality constraint.
        box_constraint: Box constraint.
        dim: Dimension of the original problem.
        dim_lifted: Dimension of the lifted problem.
        d_r: Scaling factor for the lifted dimension.

    Returns:
        ProjectionInstance: Initial state for the ADMM solver.
    """
    # Preprocess
    eq_spec = yraw.eq
    if eq_spec is not None:
        if eq_spec.a_dyn is not None:
            # Lift the equality constraint
            assert eq_spec.b is not None
            parser = ConstraintParser(
                eq_constraint=EqualityConstraint(
                    a_dyn=eq_spec.a_dyn, b=eq_spec.b, method="pinv"
                ),
                ineq_constraint=ineq_constraint,
                box_constraint=box_constraint,
            )
            lifted_eq_constraint, _, _ = parser.parse(method="pinv")
            assert lifted_eq_constraint is not None
            eq_spec = eq_spec.update(
                a_dyn=lifted_eq_constraint.a_dyn, Apinv=lifted_eq_constraint.a_dyn_pinv
            )
            yraw = yraw.update(eq=eq_spec)

        if eq_spec.b is not None:
            b_lifted = (
                jnp.concatenate(
                    [
                        eq_spec.b,
                        jnp.zeros(shape=(eq_spec.b.shape[0], dim_lifted - dim, 1)),
                    ],
                    axis=1,
                )
                * d_r
            )
            yraw = yraw.update(eq=eq_spec.update(b=b_lifted))

    # Return updated value
    return yraw.update(x=jnp.zeros((yraw.x.shape[0], dim_lifted, 1)))


def build_iteration_step(
    eq_constraint: EqualityConstraint,
    box_constraint: BoxConstraint,
    dim: int,
    scale: jnp.ndarray | float = 1.0,
) -> tuple[
    Callable[[ProjectionInstance, ProjectionInstance, float, float], ProjectionInstance],
    Callable[[ProjectionInstance], ProjectionInstance],
]:
    """Build the iteration and result retrieval step for the ADMM solver.

    See https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf for details.
    Args:
        eq_constraint: (Lifted) Equality constraint.
        box_constraint: (Lifted) Box constraint.
        dim: Dimension of the original problem.
        scale: Scaling of primal variables.

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
        yraw: ProjectionInstance,
        sigma: float = 1.0,
        omega: float = 1.7,
    ) -> ProjectionInstance:
        """One iteration of the ADMM solver.

        Args:
            sk: State iterate for the ADMM solver.
                .x of Shape (batch_size, lifted_dimension, 1).
            yraw: Point to be projected. .x of Shape (batch_size, dimension, 1).
            sigma: ADMM parameter.
            omega: ADMM parameter.

        Returns:
            ProjectionInstance: Next state iterate of the ADMM solver.
        """
        zk = eq_constraint.project(sk)
        # Reflection
        reflect = 2 * zk.x - sk.x
        tobox = jnp.concatenate(
            (
                (2 * sigma * scale * yraw.x + reflect[:, :dim, :])
                / (1 + 2 * sigma * scale**2),
                reflect[:, dim:, :],
            ),
            axis=1,
        )
        tk = box_constraint.project(sk.update(x=tobox))
        sk = sk.update(x=sk.x + omega * (tk.x - zk.x))
        return sk

    # The second element is used to extract the projection from the auxiliary
    return (iteration_step, eq_constraint.project)
