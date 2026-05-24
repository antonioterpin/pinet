"""Parser of constraints to lifted representation module."""

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from pinet._typing import ArrayLike
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
from .soc_constraint import SocConstraint

ParseLifter = Callable[[ProjectionInstance], ProjectionInstance]
ParseResult = tuple[
    EqualityConstraint | None,
    BoxConstraint | CartesianConstraint | None,
    ParseLifter,
]


class ConstraintParser:
    """Parse constraints into a lifted representation.

    This class takes as input an equality, an inequality, and a box constraint.
    It returns an equivalent equality and box constraint in a lifted representation.
    """

    def __init__(
        self,
        eq_constraint: EqualityConstraint | None = None,
        ineq_constraint: AffineInequalityConstraint | None = None,
        box_constraint: BoxConstraint | None = None,
        nl_constraints: list[NonLinearConstraint] | None = None,
    ) -> None:
        """Initialize the constraint parser.

        Args:
            eq_constraint: An equality constraint.
            ineq_constraint: An inequality constraint.
            box_constraint: A box constraint.
            nl_constraints: Non-linear constraints.
        """
        self.ineq_constraint = ineq_constraint
        self.nl_constraints = nl_constraints
        self._identity_mode = ineq_constraint is None and nl_constraints is None

        if self._identity_mode:
            # No lifting needed: parse returns the inputs as-is.
            self.eq_constraint = eq_constraint
            self.box_constraint = box_constraint
            self.dim = 0
            self.n_eq = 0
            self.n_ineq = 0
            return

        if ineq_constraint is not None:
            self.dim = ineq_constraint.dim
        else:
            assert nl_constraints is not None
            self.dim = nl_constraints[0].dim

        if eq_constraint is None:
            eq_constraint = EqualityConstraint(
                a_mat=jnp.empty((1, 0, self.dim)),
                b=jnp.empty((1, 0, 1)),
                method=None,
                var_b=False,
                var_a_mat=False,
            )

        self.eq_constraint = eq_constraint
        self.n_eq = eq_constraint.n_constraints
        self.n_ineq = ineq_constraint.n_constraints if ineq_constraint is not None else 0
        self.box_constraint = box_constraint

        if ineq_constraint is not None:
            # Batch consistency checks
            # Equality and inequality matrices must have compatible batch dimensions.
            assert (
                self.eq_constraint.a_mat.shape[0] == ineq_constraint.c_mat.shape[0]
                or self.eq_constraint.a_mat.shape[0] == 1
                or ineq_constraint.c_mat.shape[0] == 1
            ), "Batch sizes of a_mat and c_mat must be consistent."
            if box_constraint is not None:
                # An explicit box constraint must provide its lower bounds.
                assert box_constraint.lb is not None
                # An explicit box constraint must provide its upper bounds.
                assert box_constraint.ub is not None
                # Inequality lower bounds and box lower bounds must batch together.
                assert (
                    ineq_constraint.lb.shape[0] == box_constraint.lb.shape[0]
                    or ineq_constraint.lb.shape[0] == 1
                    or box_constraint.lb.shape[0] == 1
                ), "Batch sizes of lb and lower_bound must be consistent."

                # Inequality upper bounds and box upper bounds must batch together.
                assert (
                    ineq_constraint.ub.shape[0] == box_constraint.ub.shape[0]
                    or ineq_constraint.ub.shape[0] == 1
                    or box_constraint.ub.shape[0] == 1
                ), "Batch sizes of ub and upper_bound must be consistent."
        if nl_constraints is not None:
            for non_linear in nl_constraints:
                assert non_linear.a_mat is None or non_linear.a_mat.shape[0] == 1, (
                    "Batch size of non-linear constraint a_mat must be 1 or None."
                )
                assert non_linear.f is None or non_linear.f.shape[0] == 1, (
                    "Batch size of non-linear constraint f must be 1 or None."
                )
            if ineq_constraint is not None:
                assert ineq_constraint.c_mat.shape[0] == 1, (
                    "Batch size of inequality constraint must be 1 or None."
                )

    def parse(self, method: str | None = "pinv") -> ParseResult:
        """Parse the constraints into a lifted representation.

        Dispatches to the identity, polytope, or non-linear parser depending
        on the constraint configuration passed at construction time.

        Args:
            method: Method to use for solving linear systems.
                Valid methods are "pinv", and None.

        Returns:
            A tuple (eq_constraint, primitive_constraint, lift_function).
        """
        if self._identity_mode:
            return (self.eq_constraint, self.box_constraint, lambda y: y)
        if self.nl_constraints is not None:
            return self.parse_non_linear(method)
        return self.parse_polytope(method)

    def parse_polytope(
        self, method: str | None = "pinv"
    ) -> tuple[EqualityConstraint, BoxConstraint, ParseLifter]:
        """Parse equality + inequality + box constraints into lifted form.

        Args:
            method: Method to use for solving linear systems.

        Returns:
            A tuple (lifted_eq_constraint, lifted_primitive_constraint, lift_function).
        """
        ineq_constraint = self.ineq_constraint
        # Precondition: the polytope path requires an inequality constraint.
        assert ineq_constraint is not None
        eq_constraint = self.eq_constraint
        # Precondition: eq_constraint is set by __init__ for non-identity mode.
        assert eq_constraint is not None

        # Build lifted a_mat matrix.
        # Maximum batch size between a_mat and c_mat.
        mb_ac = max(eq_constraint.a_mat.shape[0], ineq_constraint.c_mat.shape[0])
        first_row_batched = jnp.tile(
            jnp.concatenate(
                [
                    eq_constraint.a_mat,
                    jnp.zeros(
                        shape=(eq_constraint.a_mat.shape[0], self.n_eq, self.n_ineq)
                    ),
                ],
                axis=2,
            ),
            (mb_ac // eq_constraint.a_mat.shape[0], 1, 1),
        )
        second_row_batched = jnp.tile(
            jnp.concatenate(
                [
                    ineq_constraint.c_mat,
                    -jnp.tile(
                        jnp.eye(self.n_ineq).reshape(1, self.n_ineq, self.n_ineq),
                        (ineq_constraint.c_mat.shape[0], 1, 1),
                    ),
                ],
                axis=2,
            ),
            (mb_ac // ineq_constraint.c_mat.shape[0], 1, 1),
        )
        a_mat_lifted = jnp.concatenate([first_row_batched, second_row_batched], axis=1)
        b_lifted = jnp.concatenate(
            [
                eq_constraint.b,
                jnp.zeros(shape=(eq_constraint.b.shape[0], self.n_ineq, 1)),
            ],
            axis=1,
        )
        eq_lifted = EqualityConstraint(
            a_mat=a_mat_lifted,
            b=b_lifted,
            method=method,
            var_b=eq_constraint.var_b,
            var_a_mat=eq_constraint.var_a_mat,
        )

        if self.box_constraint is None:
            # We only project the lifted part.
            mask_box = jnp.concatenate(
                [jnp.zeros(self.dim, dtype=bool), jnp.ones(self.n_ineq, dtype=bool)]
            )
            box_lifted = BoxConstraint(
                BoxConstraintSpecification(
                    lb=ineq_constraint.lb,
                    ub=ineq_constraint.ub,
                    mask=mask_box,
                )
            )
        else:
            # We project both the lifted and the initial box
            # The original box mask is needed to place existing bounds in the lift.
            assert self.box_constraint.mask is not None
            # The original lower bounds are needed to concatenate lifted bounds.
            assert self.box_constraint.lb is not None
            # The original upper bounds are needed to concatenate lifted bounds.
            assert self.box_constraint.ub is not None
            mask_box = jnp.concatenate(
                [
                    self.box_constraint.mask,
                    jnp.ones(self.n_ineq, dtype=bool),
                ]
            )
            # Maximum batch dimension for lower bound
            mblb = max(
                self.box_constraint.lb.shape[0],
                ineq_constraint.lb.shape[0],
            )
            lifted_lb = jnp.concatenate(
                [
                    jnp.tile(
                        self.box_constraint.lb,
                        (mblb // self.box_constraint.lb.shape[0], 1, 1),
                    ),
                    jnp.tile(
                        ineq_constraint.lb,
                        (mblb // ineq_constraint.lb.shape[0], 1, 1),
                    ),
                ],
                axis=1,
            )
            # Maximum batch dimension for upper bound
            mbub = max(
                self.box_constraint.ub.shape[0],
                ineq_constraint.ub.shape[0],
            )
            lifted_ub = jnp.concatenate(
                [
                    jnp.tile(
                        self.box_constraint.ub,
                        (mbub // self.box_constraint.ub.shape[0], 1, 1),
                    ),
                    jnp.tile(
                        ineq_constraint.ub,
                        (mbub // ineq_constraint.ub.shape[0], 1, 1),
                    ),
                ],
                axis=1,
            )
            box_lifted = BoxConstraint(
                BoxConstraintSpecification(
                    lb=lifted_lb,
                    ub=lifted_ub,
                    mask=mask_box,
                )
            )

        def lift(y: ProjectionInstance) -> ProjectionInstance:
            """Lift the input to the lifted dimension.

            Args:
                y: Projection instance to be lifted by augmenting the primal
                    variable with the inequality slack component.

            Returns:
                The lifted projection instance.
            """
            y = y.update(x=jnp.concatenate([y.x, ineq_constraint.c_mat @ y.x], axis=1))
            if eq_constraint.var_b:
                # Variable equality data must be present before extending the RHS.
                assert y.eq is not None
                # The lifted RHS can be built only if the original b is available.
                assert y.eq.b is not None
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
        self, method: str | None = "pinv"
    ) -> tuple[EqualityConstraint, CartesianConstraint, ParseLifter]:
        """Parse equality + inequality + box + non-linear constraints into lifted form.

        Args:
            method: Method to use for solving linear systems.

        Returns:
            A tuple (lifted_eq_constraint, cartesian_constraint, lift_function).
        """
        nl_constraints = self.nl_constraints
        # Precondition: the non-linear path requires non-linear constraints.
        assert nl_constraints is not None
        eq_constraint = self.eq_constraint
        # Precondition: eq_constraint is set by __init__ for non-identity mode.
        assert eq_constraint is not None

        # ``ArrayLike`` here matches the public spec annotations (jax + numpy)
        # so all the ``.append`` calls below typecheck regardless of which
        # backend the caller passed in.
        all_matrices: list[ArrayLike] = [eq_constraint.a_mat]
        dims: list[int] = [eq_constraint.dim]
        if self.ineq_constraint is not None:
            all_matrices.append(self.ineq_constraint.c_mat)
            dims.append(self.ineq_constraint.n_constraints)
        for non_linear in nl_constraints:
            # Non-linear constraints must expose their linear-transformation matrix.
            assert non_linear.a_mat is not None
            all_matrices.append(non_linear.a_mat)
            dims.append(non_linear.a_mat.shape[1])
            # Normalise the linear RHS row so SOCType (with or without ``f``)
            # and L2NormType all flow through the same lifted-auxiliary
            # plumbing: every NL constraint contributes ``m`` aux rows for
            # the ``u_aux = A x`` equality (A is ``non_linear.a_mat``) plus
            # one row for the bound. For L2NormType (constant bound) and
            # SOCType-without-f, the row is synthesised as zeros so the
            # ``t_aux`` ends up identically zero and the SOC's own ``b``
            # offset carries the original scalar. ``nl_type`` is guaranteed
            # supported by ``NonLinearSpecification._validate_type`` (run at
            # ``NonLinearConstraint`` construction), so no type guard here.
            f_row: ArrayLike = (
                non_linear.f
                if non_linear.f is not None
                else jnp.zeros((1, 1, non_linear.a_mat.shape[2]))
            )
            all_matrices.append(f_row)
            dims[-1] += 1
        # Dimension of auxiliary variables. ``dims`` is a Python list of
        # ``int`` shape values, so use the built-in ``sum`` to avoid
        # producing a traced array inside ``jnp.sum`` (which would error
        # when ``parse_non_linear`` is re-invoked from the jitted
        # ``solver.admm.initialize`` for the ``var_a_mat=True`` re-lift).
        n_aux = sum(dims[1:])
        n_tot = sum(dims)
        # Tile each row block to the largest batch size before concatenating
        # along the row axis. With ``var_a_mat=True`` the equality matrix
        # carries a per-instance batch (>1); inequality and non-linear
        # matrices are constrained to batch 1 (validated above).
        max_batch = max(M.shape[0] for M in all_matrices)
        all_matrices = [
            jnp.tile(M, (max_batch // M.shape[0], 1, 1)) for M in all_matrices
        ]
        # Build first block column of lifted A
        lifted_a_b1 = jnp.concatenate(all_matrices, axis=1)
        # Append zeros
        lifted_a_b2 = jnp.zeros((max_batch, lifted_a_b1.shape[1], n_aux))
        a_lifted = jnp.concatenate([lifted_a_b1, lifted_a_b2], axis=2)
        # Make matrix for auxiliaries
        aux_mat = jnp.zeros_like(a_lifted)
        start_row = eq_constraint.n_constraints
        start_col = eq_constraint.dim
        aux_mat = aux_mat.at[:, start_row:, start_col:].set(
            -jnp.eye(n_aux).reshape(1, n_aux, n_aux)
        )
        a_lifted = a_lifted + aux_mat
        b_lifted = jnp.concatenate(
            [eq_constraint.b, jnp.zeros(shape=(eq_constraint.b.shape[0], n_aux, 1))],
            axis=1,
        )
        # Define lifted equality constraints. ``var_a_mat`` propagates from the
        # input: when the user supplies a per-instance ``a_mat`` via
        # ``y_raw.eq.a_mat``, ``solver.admm.initialize`` re-runs
        # ``parse_non_linear`` with the new value and produces a fresh lifted
        # ``a_mat`` / ``a_mat_pinv`` whose top block reflects that input.
        eq_lifted = EqualityConstraint(
            a_mat=a_lifted,
            b=b_lifted,
            method=method,
            var_b=eq_constraint.var_b,
            var_a_mat=eq_constraint.var_a_mat,
        )
        # Setup primitive constraints -> Box, SOC, ...
        prim_constraints: list[SocConstraint] = []
        # Running dimension for convenience
        n_curr = eq_constraint.dim
        # Inequality and box constraints
        box_lifted: BoxConstraint | None = None
        if self.box_constraint is not None or self.ineq_constraint is not None:
            if self.box_constraint is not None and self.box_constraint.mask is not None:
                # Explicit box with a mask: use its masks and bounds directly.
                assert self.box_constraint.lb is not None
                assert self.box_constraint.ub is not None
                mask_box_init = self.box_constraint.mask
                box_lb_init = self.box_constraint.lb
                box_ub_init = self.box_constraint.ub
            else:
                mask_box_init = jnp.zeros(self.dim, dtype=jnp.bool_)
                box_lb_init = jnp.full((1, 0, 1), -jnp.inf)
                box_ub_init = jnp.full((1, 0, 1), jnp.inf)
            if self.ineq_constraint is not None:
                mask_box_ineq = jnp.ones(
                    self.ineq_constraint.n_constraints, dtype=jnp.bool_
                )
                box_lb_ineq = self.ineq_constraint.lb
                box_ub_ineq = self.ineq_constraint.ub
                n_curr += self.ineq_constraint.n_constraints
            else:
                mask_box_ineq = jnp.zeros(shape=(0,), dtype=jnp.bool_)
                box_lb_ineq = jnp.full((1, 0, 1), -jnp.inf)
                box_ub_ineq = jnp.full((1, 0, 1), jnp.inf)
            mask_box_other = jnp.array(
                [False] * (n_tot - mask_box_init.size - mask_box_ineq.size),
                dtype=jnp.bool_,
            )
            mask_box_lifted = jnp.concatenate(
                [mask_box_init, mask_box_ineq, mask_box_other], axis=0
            )
            box_lb_lifted = jnp.concatenate([box_lb_init, box_lb_ineq], axis=1)
            box_ub_lifted = jnp.concatenate([box_ub_init, box_ub_ineq], axis=1)
            box_lifted = BoxConstraint(
                BoxConstraintSpecification(
                    lb=box_lb_lifted,
                    ub=box_ub_lifted,
                    mask=mask_box_lifted,
                )
            )
        # Non-linear constraints. SOCType and L2NormType both lower to the
        # same SocConstraint here — the difference is whether ``t_aux`` is
        # ``f @ x`` (SOC, set by the equality block above) or constant zero
        # (L2Norm, synthesised f-row), and the ``b`` offset of the SOC
        # carries the user-supplied scalar bound either way.
        for nl in nl_constraints:
            # Non-linear constraints need a linear-transformation matrix for lifting.
            assert nl.a_mat is not None
            # ``nl_type`` is guaranteed supported by
            # ``NonLinearSpecification._validate_type`` (run at
            # ``NonLinearConstraint`` construction). SOCType and L2NormType
            # both lower to the same SocConstraint; the difference is carried
            # by the synthesised f-row in the equality block above.
            # Masks depend only on (static) dimensions, not on traced
            # tensor values. Build them as ``numpy`` arrays so the
            # downstream ``validate()`` (which calls ``.sum()``) returns
            # a concrete Python scalar — important when
            # ``parse_non_linear`` is re-invoked from the jitted
            # ``solver.admm.initialize`` for ``var_a_mat=True``.
            socspec = SocConstraintSpecification(
                mask_u=np.array(
                    [False] * n_curr
                    + [True] * nl.a_mat.shape[1]
                    + [False] * (n_tot - n_curr - nl.a_mat.shape[1]),
                    dtype=np.bool_,
                ),
                mask_t=np.array(
                    [False] * (n_curr + nl.a_mat.shape[1])
                    + [True]
                    + [False] * (n_tot - n_curr - nl.a_mat.shape[1] - 1),
                    dtype=np.bool_,
                ),
                a=nl.a,
                b=nl.b,
            )
            prim_constraints.append(SocConstraint(socspec=socspec))
            n_curr += nl.a_mat.shape[1] + 1

        cartesian_lifted = CartesianConstraint(
            box_constraint=box_lifted, nl_constraints=prim_constraints
        )

        def lift(y: ProjectionInstance) -> ProjectionInstance:
            """Lift the input to the lifted dimension.

            Args:
                y: Projection instance to be lifted by augmenting the primal
                    variable with the non-linear auxiliary slack component.

            Returns:
                The lifted projection instance.
            """
            # Build auxiliary variables
            y = y.update(
                x=jnp.concatenate(
                    [y.x, lifted_a_b1[:, eq_constraint.n_constraints :, :] @ y.x],
                    axis=1,
                )
            )
            if eq_constraint.var_b:
                # Variable equality requires the per-input spec to be present.
                assert y.eq is not None
                assert y.eq.b is not None
                y = y.update(
                    eq=y.eq.update(
                        b=jnp.concatenate(
                            [
                                y.eq.b,
                                jnp.zeros((y.x.shape[0], a_lifted.shape[2] - self.dim)),
                            ]
                        )
                    )
                )
            return y

        return (eq_lifted, cartesian_lifted, lift)
