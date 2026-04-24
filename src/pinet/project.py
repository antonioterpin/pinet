"""Implementation of the projection layer."""

from collections.abc import Callable
from functools import partial

import jax
from jax import numpy as jnp

from ._typing import BatchedScalar, ColScaling, RowScaling
from .constants import Constants
from .constraints import (
    AffineInequalityConstraint,
    BoxConstraint,
    ConstraintParser,
    EqualityConstraint,
    NonLinearConstraint,
)
from .dataclasses import (
    BoxConstraintSpecification,
    EquilibrationParams,
    ProjectionInstance,
)
from .equilibration import ruiz_equilibration
from .solver import build_iteration_step, initialize

PROJECTION_DEFAULT_SIGMA = Constants.PROJECTION_DEFAULT_SIGMA
PROJECTION_DEFAULT_OMEGA = Constants.PROJECTION_DEFAULT_OMEGA
PROJECTION_DEFAULT_CHECK_EVERY = Constants.PROJECTION_DEFAULT_CHECK_EVERY
PROJECTION_DEFAULT_TOL = Constants.PROJECTION_DEFAULT_TOL
PROJECTION_DEFAULT_MAX_ITER = Constants.PROJECTION_DEFAULT_MAX_ITER
PROJECTION_DEFAULT_CHECK_REDUCTION = Constants.PROJECTION_DEFAULT_CHECK_REDUCTION


class Project:
    """Projection layer implemented via Douglas-Rachford.

    Attributes:
        eq_constraint: Equality constraint.
        ineq_constraint: Affine inequality constraint.
        box_constraint: Box constraint.
        nl_constraints: List of non-linear constraints.
        unroll: Use loop unrolling for backpropagation.
        equilibration_params: Parameters for equilibration.
    """

    eq_constraint: EqualityConstraint | None = None
    ineq_constraint: AffineInequalityConstraint | None = None
    box_constraint: BoxConstraint | None = None
    nl_constraints: list[NonLinearConstraint] | None = None
    unroll: bool = False
    equilibration_params: EquilibrationParams | None = None

    def __init__(
        self,
        eq_constraint: EqualityConstraint | None = None,
        ineq_constraint: AffineInequalityConstraint | None = None,
        box_constraint: BoxConstraint | None = None,
        nl_constraints: list[NonLinearConstraint] | None = None,
        unroll: bool = False,
        equilibration_params: EquilibrationParams | None = None,
    ) -> None:
        """Initialize projection layer.

        Args:
            eq_constraint: Equality constraint.
            ineq_constraint: Affine inequality constraint.
            box_constraint: Box constraint.
            nl_constraints: List of non-linear constraints.
            unroll: Use loop unrolling for backpropagation.
            equilibration_params: Parameters for equilibration.
        """
        self.eq_constraint = eq_constraint
        self.ineq_constraint = ineq_constraint
        self.box_constraint = box_constraint
        self.nl_constraints = nl_constraints
        self.unroll = unroll
        if equilibration_params is None:
            self.equilibration_params = EquilibrationParams()
        else:
            self.equilibration_params = equilibration_params
        self.setup()

    def setup(self) -> None:
        """Setup the projection layer."""
        constraints = [
            c
            for c in (
                self.eq_constraint,
                self.box_constraint,
                self.ineq_constraint,
                *(self.nl_constraints or []),
            )
            if c is not None
        ]
        # The projection layer is meaningful only if at least one constraint is active.
        assert len(constraints) > 0, "At least one constraint must be provided."
        self.dim = constraints[0].dim

        is_single_simple_constraint = (
            self.ineq_constraint is None
            and self.nl_constraints is None
            and len(constraints) == 1
        )
        self.is_single_simple_constraint = is_single_simple_constraint

        self.dim_lifted = self.dim
        self.step_iteration = lambda s_prev, yraw, sigma, omega: s_prev
        self.step_final = self._project_single
        self.single_constraint = constraints[0]
        self.d_r = jnp.ones((1, self.single_constraint.n_constraints, 1))
        self.d_c = jnp.ones((1, self.single_constraint.dim, 1))
        if not self.is_single_simple_constraint:
            if self.nl_constraints is None:
                # Constraints need to be parsed
                if self.ineq_constraint is not None:
                    self.dim_lifted += self.ineq_constraint.n_constraints
                parser = ConstraintParser(
                    eq_constraint=self.eq_constraint,
                    ineq_constraint=self.ineq_constraint,
                    box_constraint=self.box_constraint,
                )
                (parsed_eq, parsed_box, self.lift) = parser.parse(method=None)
                # Inequality-constrained parsing must yield a lifted equality.
                assert parsed_eq is not None
                # Inequality-constrained parsing must yield a lifted box.
                assert parsed_box is not None
                # Setup always stores equilibration parameters before this branch runs.
                assert self.equilibration_params is not None
                # Only equilibrate when we have a single a_dyn.
                if not parsed_eq.var_a_dyn and parsed_eq.a_dyn.shape[0] == 1:
                    scaled_a_dyn_flat, d_r_flat, d_c_flat = ruiz_equilibration(
                        parsed_eq.a_dyn[0], self.equilibration_params
                    )
                    scaled_a_dyn = scaled_a_dyn_flat.reshape(
                        1, parsed_eq.a_dyn.shape[1], parsed_eq.a_dyn.shape[2]
                    )
                    self.d_r = d_r_flat.reshape(1, -1, 1)
                    self.d_c = d_c_flat.reshape(1, -1, 1)
                else:
                    # No equilibration for variable a_dyn
                    n_ineq = (
                        self.ineq_constraint.n_constraints
                        if self.ineq_constraint is not None
                        else 0
                    )
                    n_eq = (
                        self.eq_constraint.n_constraints
                        if self.eq_constraint is not None
                        else 0
                    )
                    scaled_a_dyn = parsed_eq.a_dyn
                    self.d_r = jnp.ones((1, n_eq + n_ineq, 1))
                    self.d_c = jnp.ones((1, self.dim_lifted, 1))

                # Build the scaled lifted equality constraint with method="pinv".
                self.lifted_eq_constraint = EqualityConstraint(
                    a_dyn=scaled_a_dyn,
                    b=parsed_eq.b * self.d_r,
                    method="pinv",
                    var_b=parsed_eq.var_b,
                    var_a_dyn=parsed_eq.var_a_dyn,
                )

                # Scale the lifted box constraints.
                # The polytope path always produces a BoxConstraint (not a cartesian).
                assert isinstance(parsed_box, BoxConstraint)
                # BoxConstraint.__init__ guarantees mask/lb/ub are set.
                assert parsed_box.mask is not None
                assert parsed_box.lb is not None
                assert parsed_box.ub is not None
                mask = parsed_box.mask
                box_scale = 1 / self.d_c[:, mask, :]

                self.lifted_box_constraint = BoxConstraint(
                    BoxConstraintSpecification(
                        lb=parsed_box.lb * box_scale,
                        ub=parsed_box.ub * box_scale,
                        mask=parsed_box.mask,
                    ),
                    scale=box_scale,
                )

                self.step_iteration, self.step_final = build_iteration_step(
                    self.lifted_eq_constraint,
                    self.lifted_box_constraint,
                    self.dim,
                    self.d_c[:, : self.dim, :],
                )
            else:
                # Compute lifted dimension
                if self.ineq_constraint is not None:
                    self.dim_lifted += self.ineq_constraint.n_constraints
                for nl in self.nl_constraints:
                    self.dim_lifted += nl.A.shape[1]
                    if nl.f is not None:
                        self.dim_lifted += 1

                parser = ConstraintParser(
                    eq_constraint=self.eq_constraint,
                    ineq_constraint=self.ineq_constraint,
                    box_constraint=self.box_constraint,
                    nl_constraints=self.nl_constraints,
                )
                # TODO: Change the "lifted_box_constraint" name?
                # This is cartesian constraint now.
                (
                    self.lifted_eq_constraint,
                    self.lifted_box_constraint,
                    self.lift,
                ) = parser.parse(method="pinv")
                # The non-linear path must produce a lifted equality and a cartesian.
                assert self.lifted_eq_constraint is not None
                assert self.lifted_box_constraint is not None
                # Impose no rescaling
                self.d_r = jnp.ones((1, self.lifted_eq_constraint.a_dyn.shape[1], 1))
                self.d_c = jnp.ones((1, self.dim_lifted, 1))

                self.step_iteration, self.step_final = build_iteration_step(
                    eq_constraint=self.lifted_eq_constraint,
                    box_constraint=self.lifted_box_constraint,
                    dim=self.dim,
                    scale=self.d_c[:, : self.dim, :],
                )

        if is_single_simple_constraint:
            # For a single simple constraint the projection is a closed-form
            # one-step operation: proj(x) = x - A^+ (Ax - b).
            # The ADMM initializer zeros out yraw.x, causing _project_single
            # to project the origin rather than the actual input point.
            # Override initialize so yraw.x is preserved end-to-end.
            self.initialize = lambda yraw: yraw

        project_fn = (
            _project_general
            if (self.unroll or self.is_single_simple_constraint)
            else _project_general_custom
        )

        static_args = (
            ["n_iter"]
            if (self.unroll or self.is_single_simple_constraint)
            else ["n_iter", "n_iter_bwd", "fpi"]
        )

        self._project = jax.jit(
            partial(
                project_fn,
                initialize_fn=self.initialize,
                step_iteration=self.step_iteration,
                step_final=self.step_final,
                dim_lifted=self.dim_lifted,
                d_r=self.d_r,
                d_c=self.d_c,
            ),
            static_argnames=static_args,
        )

        # jit correctly the call method
        self.call = self._project

    def initialize(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Returns a zero initial value for the governing sequence.

        Args:
            yraw: Point to be projected data.

        Returns:
            ProjectionInstance: Initial value for the governing sequence.
        """
        return initialize(
            yraw=yraw,
            ineq_constraint=self.ineq_constraint,
            box_constraint=self.box_constraint,
            dim=self.dim,
            dim_lifted=self.dim_lifted,
            d_r=self.d_r,
        )

    def cv(self, y: ProjectionInstance) -> BatchedScalar:
        """Compute the constraint violation.

        Args:
            y: Point to be evaluated.

        Returns:
            Constraint violation for each point in the batch.
        """
        if self.is_single_simple_constraint:
            return self.single_constraint.cv(y)

        # The lifted equality constraint must exist for the general projection path.
        assert self.lifted_eq_constraint is not None
        # The lifted box constraint must exist for the general projection path.
        assert self.lifted_box_constraint is not None
        if y.x.shape[1] != self.dim_lifted:
            y = self.lift(y)
        return jnp.maximum(
            self.lifted_eq_constraint.cv(y),
            self.lifted_box_constraint.cv(y),
        )

    def call_and_check(
        self,
        sigma: float = PROJECTION_DEFAULT_SIGMA,
        omega: float = PROJECTION_DEFAULT_OMEGA,
        check_every: int = PROJECTION_DEFAULT_CHECK_EVERY,
        tol: float = PROJECTION_DEFAULT_TOL,
        max_iter: int = PROJECTION_DEFAULT_MAX_ITER,
        reduction: str | float = PROJECTION_DEFAULT_CHECK_REDUCTION,
    ) -> Callable[[ProjectionInstance], tuple[ProjectionInstance, jax.Array, int]]:
        """Returns a function that projects input and checks constraint violation.

        Args:
            sigma: ADMM parameter.
            omega: ADMM parameter.
            check_every: Frequency of checking constraint violation.
            tol: Tolerance for constraint violation.
            max_iter: Maximum number of iterations for checking.
            reduction: Method to reduce constraint violations among a batch.
                Valid options are "max" (maximum cv less than tol),
                "mean" (mean cv less than tol), or a float in (0, 1)
                (fraction of instances with cv less than tol).

        Returns:
            Callable: Takes as input the points to be projected and any
                specifications for the constraints (e.g., the value of b for
                variable b equality constraints). Returns an approximately
                projected point and a flag showing whether the termination
                condition was satisfied.
        """

        @jax.jit
        def check(inp: ProjectionInstance) -> jax.Array:
            if reduction == "max":
                return jnp.max(self.cv(inp)) < tol
            elif reduction == "mean":
                return jnp.mean(self.cv(inp)) < tol
            elif isinstance(reduction, float) and 0 < reduction < 1:
                return jnp.mean(self.cv(inp) < tol) >= reduction
            else:
                raise ValueError(
                    f"Invalid reduction method {reduction}. "
                    "Valid options are: 'max', 'mean', or a float in (0, 1)."
                )

        def project_and_check(
            yraw: ProjectionInstance,
        ) -> tuple[ProjectionInstance, jax.Array, int]:
            # Executed iterations
            iter_exec = 0
            terminate = False
            # Call the projection function with all given arguments.
            y0 = self.initialize(yraw)
            xproj = yraw
            while not (terminate or iter_exec >= max_iter):
                xproj, y = self.call(
                    s0=y0,
                    yraw=yraw,
                    sigma=sigma,
                    omega=omega,
                    n_iter=check_every,
                )
                y0 = y
                iter_exec += check_every
                terminate = check(xproj)

            return xproj, jnp.array(terminate), iter_exec

        return project_and_check

    def _project_single(self, yraw: ProjectionInstance) -> ProjectionInstance:
        """Project a batch of points with single constraint.

        Args:
            yraw: Point to be projected.
                Shape (batch_size, dimension, 1).

        Returns:
            ProjectionInstance: The projected point for each point in the batch.
        """
        if yraw.eq and yraw.eq.a_dyn is not None:
            a_dyn_pinv = jnp.linalg.pinv(yraw.eq.a_dyn)
            yraw = yraw.update(eq=yraw.eq.update(a_dyn_pinv=a_dyn_pinv))

        return self.single_constraint.project(yraw)


# Project general
def _project_general(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, ProjectionInstance, float, float], ProjectionInstance
    ],
    step_final: Callable[[ProjectionInstance], ProjectionInstance],
    dim_lifted: int,
    d_r: RowScaling,
    d_c: ColScaling,
    yraw: ProjectionInstance,
    s0: ProjectionInstance | None = None,
    sigma: float = PROJECTION_DEFAULT_SIGMA,
    omega: float = PROJECTION_DEFAULT_OMEGA,
    n_iter: int = 100,
) -> tuple[ProjectionInstance, ProjectionInstance]:
    """Project a batch of points using Douglas-Rachford.

    Args:
        initialize_fn: Function to initialize the governing sequence.
        step_iteration: Function for the iteration step.
        step_final: Function for the final step.
        dim_lifted: Dimension of the lifted space.
        d_r: Scaling factor for the rows.
        d_c: Scaling factor for the columns.
        yraw: Point to be projected.
        s0: Initial value for the governing sequence.
        sigma: ADMM parameter.
        omega: ADMM parameter.
        n_iter: Number of iterations to run.

    Returns:
        A pair ``(projected_point, governing_sequence_value)``.
    """
    assert n_iter > 0, "Number of iterations must be positive."

    s0 = initialize_fn(yraw) if s0 is None else s0
    sk, _ = jax.lax.scan(
        lambda s_prev, _: (
            step_iteration(s_prev, yraw, sigma, omega),
            None,
        ),
        s0,
        None,
        length=n_iter,
    )

    y = step_final(sk).x[:, : yraw.x.shape[1], :]
    y_scaled = y * d_c[:, : yraw.x.shape[1], :]

    # Unscale the output
    return yraw.update(x=y_scaled), sk


@partial(
    jax.custom_vjp,
    nondiff_argnames=[
        "initialize_fn",
        "step_iteration",
        "step_final",
        "dim_lifted",
        "n_iter",
        "n_iter_bwd",
        "fpi",
    ],
)
def _project_general_custom(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, ProjectionInstance, float, float], ProjectionInstance
    ],
    step_final: Callable[[ProjectionInstance], ProjectionInstance],
    dim_lifted: int,
    d_r: RowScaling,
    d_c: ColScaling,
    yraw: ProjectionInstance,
    s0: ProjectionInstance | None = None,
    sigma: float = PROJECTION_DEFAULT_SIGMA,
    omega: float = PROJECTION_DEFAULT_OMEGA,
    n_iter: int = 0,
    n_iter_bwd: int = 5,
    fpi: bool = False,
) -> tuple[ProjectionInstance, ProjectionInstance]:
    return _project_general(
        initialize_fn=initialize_fn,
        step_iteration=step_iteration,
        step_final=step_final,
        dim_lifted=dim_lifted,
        d_r=d_r,
        d_c=d_c,
        s0=s0,
        yraw=yraw,
        sigma=sigma,
        omega=omega,
        n_iter=n_iter,
    )


def _project_general_fwd(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, ProjectionInstance, float, float], ProjectionInstance
    ],
    step_final: Callable[[ProjectionInstance], ProjectionInstance],
    dim_lifted: int,
    d_r: RowScaling,
    d_c: ColScaling,
    yraw: ProjectionInstance,
    s0: ProjectionInstance | None = None,
    sigma: float = PROJECTION_DEFAULT_SIGMA,
    omega: float = PROJECTION_DEFAULT_OMEGA,
    n_iter: int = 0,
    n_iter_bwd: int = 5,
    fpi: bool = False,
) -> tuple[
    tuple[ProjectionInstance, ProjectionInstance],
    tuple[ProjectionInstance, ProjectionInstance, RowScaling, ColScaling, float, float],
]:
    # unpack trailing options that belong only to custom vjp
    # The decorated function returns a (ProjectionInstance, ProjectionInstance) tuple,
    # but jax.custom_vjp's wrapper hides the precise signature from the typechecker.
    custom_result: tuple[ProjectionInstance, ProjectionInstance] = (
        _project_general_custom(
            initialize_fn=initialize_fn,
            step_iteration=step_iteration,
            step_final=step_final,
            dim_lifted=dim_lifted,
            d_r=d_r,
            d_c=d_c,
            s0=s0,
            yraw=yraw,
            sigma=sigma,
            omega=omega,
            n_iter=n_iter,
        )
    )
    y, s_k = custom_result

    return (y, s_k), (s_k, yraw, d_r, d_c, sigma, omega)


def _project_general_bwd(
    initialize_fn: Callable[[ProjectionInstance], ProjectionInstance],
    step_iteration: Callable[
        [ProjectionInstance, ProjectionInstance, float, float], ProjectionInstance
    ],
    step_final: Callable[[ProjectionInstance], ProjectionInstance],
    dim_lifted: int,
    n_iter: int,
    n_iter_bwd: int,
    fpi: bool,
    residuals: tuple[
        ProjectionInstance,
        ProjectionInstance,
        RowScaling,
        ColScaling,
        float,
        float,
    ],
    cotangent: tuple[ProjectionInstance, ProjectionInstance],
) -> tuple[None, None, ProjectionInstance, None, None, None]:
    """Backward pass for custom vjp.

    This function computes the vjp for the projection using the
    implicit function theorem.
    Note that, the arguments are:
    (i) any arguments for the
    forward that are not arrays;
    (ii) residuals: tuple with auxiliary data from the forward pass;
    (iii) cotangent: incoming cotangents.
    The function returns a tuple where each element corresponds
    to an array from the input.

    Args:
        initialize_fn: Function to initialize the governing sequence.
        step_iteration: Function for the iteration step.
        step_final: Function for the final step.
        dim_lifted: Dimension of the lifted space.
        n_iter: Number of iterations to run.
        n_iter_bwd: Number of iterations for backward pass.
        fpi: Whether to use fixed-point iteration.
        residuals: Auxiliary data from the forward pass.
        cotangent: Incoming cotangents.

    Returns:
        tuple: The computed cotangent for the projection.
    """
    s_k, yraw, _, d_c, sigma, omega = residuals
    cotangent_zk1, _ = cotangent

    _, iteration_vjp = jax.vjp(
        lambda xx: step_iteration(xx, yraw, sigma, omega),
        s_k,
    )
    _, iteration_vjp2 = jax.vjp(lambda xx: step_iteration(s_k, xx, sigma, omega), yraw)
    _, equality_vjp = jax.vjp(step_final, s_k)

    # Rescale the gradient
    cotangent_zk1 = cotangent_zk1.x * d_c[:, : yraw.x.shape[1], :]

    # Compute VJP of cotangent with projection before auxiliary
    cotangent_eq_6 = equality_vjp(
        s_k.update(
            x=jnp.concatenate(
                [
                    cotangent_zk1,
                    jnp.zeros(
                        (cotangent_zk1.shape[0], dim_lifted - cotangent_zk1.shape[1], 1)
                    ),
                ],
                axis=1,
            )
        )
    )[0].x
    # Run iteration
    if fpi:

        def body_fn(x, _):
            vjp = iteration_vjp(x)[0].x
            return s_k.update(x=(vjp + cotangent_eq_6)), None

        cotangent_eq_7, _ = jax.lax.scan(
            body_fn,
            s_k.update(x=jnp.zeros((cotangent_zk1.shape[0], dim_lifted, 1))),
            None,
            length=n_iter_bwd,
        )
    else:
        cotangent_eq_7 = jax.scipy.sparse.linalg.bicgstab(
            lambda x: x - iteration_vjp(s_k.update(x=x))[0].x,
            cotangent_eq_6,
            maxiter=n_iter_bwd,
        )[0]
        cotangent_eq_7 = s_k.update(x=cotangent_eq_7)

    thevjp = iteration_vjp2(cotangent_eq_7)[0]

    return (None, None, thevjp, None, None, None)


_project_general_custom.defvjp(_project_general_fwd, _project_general_bwd)
