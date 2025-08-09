"""Parser of constraints to lifted representation module."""

import jax.numpy as jnp
import numpy as np

from pinet.dataclasses import BoxConstraintSpecification

from .affine_equality import EqualityConstraint
from .affine_inequality import AffineInequalityConstraint
from .box import BoxConstraint


class ConstraintParser:
    """Parse constraints into a lifted representation.

    This class takes as input an equality, an inequality, and a box constraint.
    It returns an equivalent equality and box constraint in a lifted representation.
    """

    def __init__(
        self,
        eq_constraint: EqualityConstraint,
        ineq_constraint: AffineInequalityConstraint,
        box_constraint: BoxConstraint = None,
    ) -> None:
        """Initiaze the constraint parser.

        Args:
            eq_constraint (EqualityConstraint): An equality constraint.
            ineq_constraint (AffineInequalityConstraint): An inequality constraint.
            box_constraint (BoxConstraint): A box constraint.
        """
        if ineq_constraint is None:
            # The constraints do not need lifting.
            self.parse = lambda method: (eq_constraint, box_constraint)
            return

        self.dim = ineq_constraint.dim
        if eq_constraint is None:
            eq_constraint = EqualityConstraint(
                A=jnp.empty((1, 0, self.dim)),
                b=jnp.empty((1, 0, 1)),
                method=None,
                var_b=False,
                var_A=False,
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
                self.ineq_constraint.lb.shape[0] == self.box_constraint.lb.shape[0]
                or self.ineq_constraint.lb.shape[0] == 1
                or self.box_constraint.lb.shape[0] == 1
            ), "Batch sizes of lb and lower_bound must be consistent."

            assert (
                self.ineq_constraint.ub.shape[0] == self.box_constraint.ub.shape[0]
                or self.ineq_constraint.ub.shape[0] == 1
                or self.box_constraint.ub.shape[0] == 1
            ), "Batch sizes of ub and upper_bound must be consistent."

    def parse(self, method: str = "pinv") -> tuple[EqualityConstraint, BoxConstraint]:
        """Parse the constraints into a lifted representation.

        Args:
            method: A string that specifies the method used to solve
                linear systems. Valid method "pinv", and None.

        Returns:
            A tuple of constraints: (eq_constraint, box_constraint)
        """
        # Build lifted A matrix.
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
        eq_lifted = EqualityConstraint(
            A=A_lifted,
            b=b_lifted,
            method=method,
            var_b=self.eq_constraint.var_b,
            var_A=self.eq_constraint.var_A,
        )

        if self.box_constraint is None:
            # We only project the lifted part.
            box_mask = np.concatenate(
                [np.zeros(self.dim, dtype=bool), np.ones(self.n_ineq, dtype=bool)]
            )
            box_lifted = BoxConstraint(
                BoxConstraintSpecification(
                    lb=self.ineq_constraint.lb,
                    ub=self.ineq_constraint.ub,
                    mask=box_mask,
                )
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
                self.box_constraint.lb.shape[0],
                self.ineq_constraint.lb.shape[0],
            )
            lifted_lb = jnp.concatenate(
                [
                    jnp.tile(
                        self.box_constraint.lb,
                        (mblb // self.box_constraint.lb.shape[0], 1, 1),
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
                self.box_constraint.ub.shape[0],
                self.ineq_constraint.ub.shape[0],
            )
            lifted_ub = jnp.concatenate(
                [
                    jnp.tile(
                        self.box_constraint.ub,
                        (mbub // self.box_constraint.ub.shape[0], 1, 1),
                    ),
                    jnp.tile(
                        self.ineq_constraint.ub,
                        (mbub // self.ineq_constraint.ub.shape[0], 1, 1),
                    ),
                ],
                axis=1,
            )
            box_lifted = BoxConstraint(
                BoxConstraintSpecification(
                    lb=lifted_lb,
                    ub=lifted_ub,
                    mask=box_mask,
                )
            )
        return (eq_lifted, box_lifted)
