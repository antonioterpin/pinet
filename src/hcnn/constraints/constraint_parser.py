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
        box_constraint: BoxConstraint = None,
    ):
        """Initiaze the constraint parser.

        Args:
            eq_constraint: An equality constraint.
            ineq_constraint: An inequality constraint.
            box_constraint: A box constraint.
        """
        if ineq_constraint is None:
            # The constraints do not need lifting.
            self.parse = lambda _: (eq_constraint, box_constraint)
            return

        self.dim = ineq_constraint.dim
        if eq_constraint is None:
            eq_constraint = EqualityConstraint(
                A=jnp.empty((1, 0, self.dim)),
                b=jnp.empty((1, 0, 1)),
                method=None,
            )

        self.eq_constraint = eq_constraint
        self.n_eq = eq_constraint.n_constraints
        self.ineq_constraint = ineq_constraint
        self.n_ineq = ineq_constraint.n_constraints
        self.box_constraint = box_constraint

        # Batch consistency checks
        assert (
            self.eq_constraint.A.shape[0] == self.ineq_constraint.C.shape[0]
            or self.eq_constraint.A.shape[0] == 1
            or self.ineq_constraint.C.shape[0] == 1
        ), "Batch sizes of A and C must be consistent."
        if self.box_constraint is not None:
            assert (
                self.ineq_constraint.lb.shape[0]
                == self.box_constraint.lower_bound.shape[0]
                or self.ineq_constraint.lb.shape[0] == 1
                or self.box_constraint.lower_bound.shape[0] == 1
            ), "Batch sizes of lb and lower_bound must be consistent."

            assert (
                self.ineq_constraint.ub.shape[0]
                == self.box_constraint.upper_bound.shape[0]
                or self.ineq_constraint.ub.shape[0] == 1
                or self.box_constraint.upper_bound.shape[0] == 1
            ), "Batch sizes of ub and upper_bound must be consistent."

    def parse(self, method: str = "pinv") -> Tuple[EqualityConstraint, BoxConstraint]:
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
        # Maximum batch size between A and C
        mbAC = max(self.eq_constraint.A.shape[0], self.ineq_constraint.C.shape[0])
        first_row_batched = jnp.tile(
            jnp.concatenate(
                [
                    self.eq_constraint.A,
                    jnp.zeros(
                        shape=(self.eq_constraint.A.shape[0], self.n_eq, self.n_ineq)
                    ),
                ],
                axis=2,
            ),
            (mbAC // self.eq_constraint.A.shape[0], 1, 1),
        )
        second_row_batched = jnp.tile(
            jnp.concatenate(
                [
                    self.ineq_constraint.C,
                    -jnp.tile(
                        jnp.eye(self.n_ineq).reshape(1, self.n_ineq, self.n_ineq),
                        (self.ineq_constraint.C.shape[0], 1, 1),
                    ),
                ],
                axis=2,
            ),
            (mbAC // self.ineq_constraint.C.shape[0], 1, 1),
        )
        A_lifted = jnp.concatenate([first_row_batched, second_row_batched], axis=1)
        b_lifted = jnp.concatenate(
            [
                self.eq_constraint.b,
                jnp.zeros(shape=(self.eq_constraint.b.shape[0], self.n_ineq, 1)),
            ],
            axis=1,
        )
        eq_lifted = EqualityConstraint(A=A_lifted, b=b_lifted, method=method)
        # TODO: Memory management? After building the lifted
        # matrix we could probably discard the original one.

        if self.box_constraint is None:
            # We only project the lifted part.
            box_mask = jnp.concatenate(
                [jnp.zeros(self.dim, dtype=bool), jnp.ones(self.n_ineq, dtype=bool)]
            )
            box_lifted = BoxConstraint(
                lower_bound=self.ineq_constraint.lb,
                upper_bound=self.ineq_constraint.ub,
                mask=box_mask,
            )
        else:
            # We project both the lifted and the initial box
            box_mask = jnp.concatenate(
                [
                    self.box_constraint.mask,
                    jnp.ones(self.n_ineq, dtype=bool),
                ]
            )
            # Maximum batch dimension for lower bound
            mblb = max(
                self.box_constraint.lower_bound.shape[0],
                self.ineq_constraint.lb.shape[0],
            )
            lifted_lb = jnp.concatenate(
                [
                    jnp.tile(
                        self.box_constraint.lower_bound,
                        (mblb // self.box_constraint.lower_bound.shape[0], 1, 1),
                    ),
                    jnp.tile(
                        self.ineq_constraint.lb,
                        (mblb // self.ineq_constraint.lb.shape[0], 1, 1),
                    ),
                ],
                axis=1,
            )
            # Maximum batch dimension for upper bound
            mbub = max(
                self.box_constraint.upper_bound.shape[0],
                self.ineq_constraint.ub.shape[0],
            )
            lifted_ub = jnp.concatenate(
                [
                    jnp.tile(
                        self.box_constraint.upper_bound,
                        (mbub // self.box_constraint.upper_bound.shape[0], 1, 1),
                    ),
                    jnp.tile(
                        self.ineq_constraint.ub,
                        (mbub // self.ineq_constraint.ub.shape[0], 1, 1),
                    ),
                ],
                axis=1,
            )
            box_lifted = BoxConstraint(
                lower_bound=lifted_lb,
                upper_bound=lifted_ub,
                mask=box_mask,
            )
        return (eq_lifted, box_lifted)
