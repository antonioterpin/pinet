"""Module for the Alternating Direction Method of Multipliers (ADMM) solver."""

from collections.abc import Callable

import jax.numpy as jnp

from pinet._typing import ColScaling, RowScaling, ScalarLike
from pinet.constants import Constants
from pinet.constraints import (
    AffineInequalityConstraint,
    BoxConstraint,
    CartesianConstraint,
    ConstraintParser,
    EqualityConstraint,
    NonLinearConstraint,
)
from pinet.dataclasses import ProjectionInstance

PROJECTION_DEFAULT_SIGMA = Constants.PROJECTION_DEFAULT_SIGMA
PROJECTION_DEFAULT_OMEGA = Constants.PROJECTION_DEFAULT_OMEGA


def initialize(
    yraw: ProjectionInstance,
    ineq_constraint: AffineInequalityConstraint | None,
    box_constraint: BoxConstraint | None,
    dim: int,
    dim_lifted: int,
    d_r: RowScaling,
    nl_constraints: list[NonLinearConstraint] | None = None,
) -> ProjectionInstance:
    """Initialize the ADMM solver state.

    Args:
        yraw: Point to be projected.
        ineq_constraint: Inequality constraint.
        box_constraint: Box constraint.
        dim: Dimension of the original problem.
        dim_lifted: Dimension of the lifted problem.
        d_r: Scaling factor for the lifted dimension.
        nl_constraints: Non-linear constraints, when present. Required for the
            ``var_a_mat=True`` re-lift to route through ``parse_non_linear``
            instead of ``parse_polytope``.

    Returns:
        Initial state for the ADMM solver.
    """
    # Preprocess
    eq_spec = yraw.eq
    # ``a_mat`` is only valid alongside ``b`` (enforced by the spec
    # validators), so checking ``b`` covers both per-instance overrides.
    if eq_spec is not None and eq_spec.b is not None:
        # Lift ``b`` (zero-pad to the lifted dimension and apply the row
        # scaling) and ``a_mat`` together so the spec is updated atomically:
        # splitting this into two ``update`` calls would briefly leave the
        # spec with mismatched ``m`` axes, which beartype's runtime check
        # rejects.
        updates: dict[str, object] = {
            "b": jnp.concatenate(
                [
                    eq_spec.b,
                    jnp.zeros(shape=(eq_spec.b.shape[0], dim_lifted - dim, 1)),
                ],
                axis=1,
            )
            * d_r,
        }
        if eq_spec.a_mat is not None:
            parser = ConstraintParser(
                eq_constraint=EqualityConstraint(
                    a_mat=eq_spec.a_mat, b=eq_spec.b, method="pinv"
                ),
                ineq_constraint=ineq_constraint,
                box_constraint=box_constraint,
                nl_constraints=nl_constraints,
            )
            lifted_eq_constraint, _, _ = parser.parse(method="pinv")
            # Parsing must return the lifted equality constraint in this branch.
            assert lifted_eq_constraint is not None
            updates["a_mat"] = lifted_eq_constraint.a_mat
            updates["a_mat_pinv"] = lifted_eq_constraint.a_mat_pinv
        yraw = yraw.update(eq=eq_spec.update(**updates))

    # Return updated value
    return yraw.update(x=jnp.zeros((yraw.x.shape[0], dim_lifted, 1)))


def build_iteration_step(
    eq_constraint: EqualityConstraint,
    box_constraint: BoxConstraint | CartesianConstraint,
    dim: int,
    scale: ColScaling | float = 1.0,
) -> tuple[
    Callable[
        [ProjectionInstance, ProjectionInstance, ScalarLike, ScalarLike],
        ProjectionInstance,
    ],
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
        A pair ``(iteration_step, result_retrieval_step)``.
    """

    def iteration_step(
        sk: ProjectionInstance,
        yraw: ProjectionInstance,
        sigma: ScalarLike = PROJECTION_DEFAULT_SIGMA,
        omega: ScalarLike = PROJECTION_DEFAULT_OMEGA,
    ) -> ProjectionInstance:
        """One iteration of the ADMM solver.

        Args:
            sk: State iterate for the ADMM solver.
            yraw: Point to be projected.
            sigma: ADMM parameter.
            omega: ADMM parameter.

        Returns:
            Next state iterate of the ADMM solver.
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

    return (iteration_step, eq_constraint.project)
