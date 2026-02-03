"""Parser of constraints to lifted representation module."""

from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np

from pinet.dataclasses import (
    BoxConstraintSpecification,
    ProjectionInstance,
    SocConstraintSpecification,
)

from .affine_equality import EqualityConstraint
from .affine_inequality import AffineInequalityConstraint
from .box import BoxConstraint
from .cartesian_constraint import CartesianConstraint
from .non_linear import NonLinearConstraint
from .non_linear_types import L2NormType, SOCType
from .soc_constraint import SocConstraint

# TODO: If we only have 2 constraints where one is equality and the
# other is primitive, then we directly use these.


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
        nl_constraints: Optional[list[NonLinearConstraint]] = None,
    ) -> None:
        """Initialize the constraint parser.

        Args:
            eq_constraint (EqualityConstraint): An equality constraint.
            ineq_constraint (AffineInequalityConstraint): An inequality constraint.
            box_constraint (BoxConstraint): A box constraint.
            nl_constraints (Optional[list[NonLinearConstraint]]): Non-linear constraints.
        """
        if ineq_constraint is None and nl_constraints is None:
            # The constraints do not need lifting.
            self.parse = lambda method: (eq_constraint, box_constraint, lambda y: y)
            return

        self.dim = (
            ineq_constraint.dim
            if ineq_constraint is not None
            else nl_constraints[0].dim
        )
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
        self.nl_constraints = nl_constraints

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
        if self.nl_constraints is not None:
            for non_linear in self.nl_constraints:
                # Check that all batch sizes of matrices are 1
                assert (
                    non_linear.A.shape[0] == 1 or non_linear.A is None
                ), "Batch size of non-linear constraint A must be 1 or None."
                assert (
                    non_linear.f.shape[0] == 1 or non_linear.f is None
                ), "Batch size of non-linear constraint f must be 1 or None."
            if self.ineq_constraint is not None:
                assert (
                    self.ineq_constraint.C.shape[0] == 1
                ), "Batch size of inequality constraint C must be 1 or None."

            self.parse = self.parse_non_linear
        else:
            self.parse = self.parse_polytope

    def parse_polytope(
        self, method: Optional[str] = "pinv"
    ) -> tuple[EqualityConstraint, BoxConstraint, Callable]:
        """Parse the constraints into a lifted representation.

        Args:
            method (Optional[str]): Method to use for solving linear systems.
                Valid methods are "pinv", and None.

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

        def lift(y):
            """Lift the input to the lifted dimension."""
            y = y.update(x=jnp.concatenate([y.x, self.ineq_constraint.C @ y.x], axis=1))
            if self.eq_constraint.var_b:
                y = y.update(
                    eq=y.eq.update(
                        b=jnp.concatenate(
                            [y.eq.b, jnp.zeros((y.x.shape[0], self.n_ineq, 1))],
                            axis=1,
                        )
                    )
                )
            return y

        return (eq_lifted, box_lifted, lift)

    def parse_non_linear(
        self, method: Optional[str] = "pinv"
    ) -> tuple[EqualityConstraint, CartesianConstraint, Callable]:
        """Parse the constraints into a lifted representation.

        Returns:
            A tuple of constraints: (eq_constraint, cartesian_constraint, lift_function)
        """
        all_matrices = []
        dims = []
        if self.eq_constraint is not None:
            all_matrices.append(self.eq_constraint.A)
            dims.append(self.eq_constraint.dim)
        if self.ineq_constraint is not None:
            all_matrices.append(self.ineq_constraint.C)
            dims.append(self.ineq_constraint.n_constraints)
        for non_linear in self.nl_constraints:
            all_matrices.append(non_linear.A)
            dims.append(non_linear.A.shape[1])
            if non_linear.f is not None:
                all_matrices.append(non_linear.f)
                dims[-1] += 1
        # Dimension of auxiliary variables
        n_aux = int(jnp.sum(jnp.array(dims[1:])))
        n_tot = int(jnp.sum(jnp.array(dims)))
        # Build first block column of lifted A
        lifted_A_b1 = jnp.concatenate(all_matrices, axis=1)
        # Append zeros
        lifted_A_b2 = jnp.zeros((1, lifted_A_b1.shape[1], n_aux))
        A_lifted = jnp.concatenate([lifted_A_b1, lifted_A_b2], axis=2)
        # Make matrix for auxiliaries
        aux_mat = jnp.zeros_like(A_lifted)
        start_row = self.eq_constraint.n_constraints
        start_col = self.eq_constraint.dim
        aux_mat = aux_mat.at[:, start_row:, start_col:].set(
            -jnp.eye(n_aux).reshape(1, n_aux, n_aux)
        )
        A_lifted = A_lifted + aux_mat
        b_lifted = jnp.concatenate(
            [self.eq_constraint.b, jnp.zeros(shape=(1, n_aux, 1))],
            axis=1,
        )
        # Define lifted equality constraints
        eq_lifted = EqualityConstraint(
            A=A_lifted,
            b=b_lifted,
            method=method,
            var_b=self.eq_constraint.var_b,
            var_A=False,  # For now var_A is not supported
        )
        # Setup primitive constraints -> Box, SOC, ...
        prim_constraints = []
        # Running dimension for convenience
        n_curr = self.eq_constraint.dim if self.eq_constraint is not None else 0
        # Inequality and box constraints
        if self.box_constraint is not None or self.ineq_constraint is not None:
            if self.box_constraint.mask is not None:
                box_mask_init = self.box_constraint.mask
                box_lb_init = self.box_constraint.lb
                box_ub_init = self.box_constraint.ub
            else:
                box_mask_init = jnp.zeros(self.dim, dtype=jnp.bool_)
                box_lb_init = jnp.full((1, 0, 1), -jnp.inf)
                box_ub_init = jnp.full((1, 0, 1), jnp.inf)
            if self.ineq_constraint is not None:
                box_mask_ineq = jnp.ones(
                    self.ineq_constraint.n_constraints, dtype=jnp.bool_
                )
                box_lb_ineq = self.ineq_constraint.lb
                box_ub_ineq = self.ineq_constraint.ub
                n_curr += self.ineq_constraint.n_constraints
            else:
                box_mask_ineq = jnp.zeros(shape=(0,), dtype=jnp.bool_)
                box_lb_ineq = jnp.full((1, 0, 1), -jnp.inf)
                box_ub_ineq = jnp.full((1, 0, 1), jnp.inf)
            box_mask_other = jnp.array(
                [False] * (n_tot - box_mask_init.size - box_mask_ineq.size),
                dtype=jnp.bool_,
            )
            box_mask_lifted = jnp.concatenate(
                [box_mask_init, box_mask_ineq, box_mask_other], axis=0
            )
            box_lb_lifted = jnp.concatenate([box_lb_init, box_lb_ineq], axis=1)
            box_ub_lifted = jnp.concatenate([box_ub_init, box_ub_ineq], axis=1)
            box_lifted = BoxConstraint(
                BoxConstraintSpecification(
                    lb=box_lb_lifted,
                    ub=box_ub_lifted,
                    mask=box_mask_lifted,
                )
            )
        # Non-linear constraints
        for nl in self.nl_constraints:
            if nl.nl_type == SOCType:
                socspec = SocConstraintSpecification(
                    mask_u=jnp.array(
                        [False] * n_curr
                        + [True] * nl.A.shape[1]
                        + [False] * (n_tot - n_curr - nl.A.shape[1]),
                        dtype=jnp.bool_,
                    ),
                    mask_t=jnp.array(
                        [False] * (n_curr + nl.A.shape[1])
                        + [True]
                        + [False] * (n_tot - n_curr - nl.A.shape[1] - 1),
                        dtype=jnp.bool_,
                    ),
                    a=nl.a,
                    b=nl.b,
                )
                prim_constraints.append(SocConstraint(socspec=socspec))
                n_curr += nl.A.shape[1] + 1
            elif nl.nl_type == L2NormType:
                raise NotImplementedError("L2NormType is not implemented yet.")
            else:
                raise ValueError(
                    f"Unsupported non-linear constraint type: {type(nl.nl_type)}"
                )

        cartesian_lifted = CartesianConstraint(
            box_constraint=box_lifted, nl_constraints=prim_constraints
        )

        def lift(y: ProjectionInstance):
            """Lift the input to the lifted dimension."""
            # Build auxiliary variables
            y = y.update(x=lifted_A_b1 @ y.x)
            if self.eq_constraint.var_b:
                y = y.update(
                    eq=y.eq.update(
                        b=jnp.concatenate(
                            [
                                y.eq.b,
                                jnp.zeros((y.x.shape[0], A_lifted.shape[2] - self.dim)),
                            ]
                        )
                    )
                )

        return (eq_lifted, cartesian_lifted, lift)
