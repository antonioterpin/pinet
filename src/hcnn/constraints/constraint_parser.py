# For now, this should receive an equality and an inequality
# constraint (without box component) and return
# an equality and a box constraint.
"""Parser of constraints to lifted representation module."""
from typing import Tuple

import jax.numpy as jnp

from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.constraints.box import BoxConstraint


class ConstraintParser:
    """Parse constraints into a lifted representation.

    This class takes as input an equality, an inequality,
    and optionally a box constraint.
    It returns an equivalent equality and box constraint
    in a lifted representation.
    """

    def __init__(
        self,
        eq_constraint: EqualityConstraint,
        ineq_constraint: AffineInequalityConstraint,
    ):
        """Initiaze the constraint parser.

        Args:
            eq_constraint: An equality constraint.
            ineq_constraint: An inequality constraint.
        """
        self.eq_constraint = eq_constraint
        self.dim = eq_constraint.A.shape[2]
        self.n_eq = eq_constraint.A.shape[1]
        self.ineq_constraint = ineq_constraint
        self.n_ineq = ineq_constraint.C.shape[1]

    def parse(self, method="pinv") -> Tuple[EqualityConstraint, BoxConstraint]:
        """Parse the constraints into a lifted representation.

        Args:
            method: A string that specifies the method used to solve
                linear systems. Valid method "pinv", "cholesky".

        Returns:
            A tuple of constraints: (eq_constraint, box_constraint)
        """
        # TODO: The equality constraint given to the parser will
        # have pinved/factored A. This should only be done for the lifted here.
        # Build lifted A matrix.
        # TODO: Be careful of batch sizes. Have tests for this
        A_lifted = jnp.block(
            [
                [self.eq_constraint.A, jnp.zeros(shape=(1, self.n_eq, self.n_ineq))],
                [
                    self.ineq_constraint.C,
                    -jnp.expand_dims(jnp.eye(self.n_ineq), axis=0),
                ],
            ]
        )
        b_lifted = jnp.concatenate(
            [self.eq_constraint.b, jnp.zeros(shape=(1, self.n_ineq, 1))], axis=1
        )
        eq_lifted = EqualityConstraint(A=A_lifted, b=b_lifted, method=method)
        # TODO: Memory management? After building the lifted
        # matrix we could probably discard the original one.

        # We only project the lifted part.
        box_mask = jnp.concatenate(
            [jnp.zeros(self.dim, dtype=bool), jnp.ones(self.n_ineq, dtype=bool)]
        )
        box_lifted = BoxConstraint(
            lower_bound=self.ineq_constraint.lb,
            upper_bound=self.ineq_constraint.ub,
            mask=box_mask,
        )
        return (eq_lifted, box_lifted)
