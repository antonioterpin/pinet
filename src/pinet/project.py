"""Implementation of the projection layer."""

from functools import partial
from typing import Callable

import jax
from jax import numpy as jnp

from .constraints import (
    AffineInequalityConstraint,
    BoxConstraint,
    ConstraintParser,
    EqualityConstraint,
)
from .dataclasses import EquilibrationParams, ProjectionInstance
from .equilibration import ruiz_equilibration
from .solver import build_iteration_step


class Project:
    """Projection layer implemented via Douglas-Rachford."""

    eq_constraint: EqualityConstraint = None
    ineq_constraint: AffineInequalityConstraint = None
    box_constraint: BoxConstraint = None
    unroll: bool = False

    def __init__(
        self,
        eq_constraint: EqualityConstraint = None,
        ineq_constraint: AffineInequalityConstraint = None,
        box_constraint: BoxConstraint = None,
        unroll: bool = False,
        equilibration_params: EquilibrationParams = EquilibrationParams(),
    ) -> None:
        """Initialize projection layer.

        Args:
            eq_constraint (EqualityConstraint): Equality constraint.
            ineq_constraint (AffineInequalityConstraint): Inequality constraint.
            box_constraint (BoxConstraint): Box constraint.
            unroll (bool): Use loop unrolling for backpropagation.
            equilibration_params (EquilibrationParams): Parameters for equilibration.
        """
        self.eq_constraint = eq_constraint
        self.ineq_constraint = ineq_constraint
        self.box_constraint = box_constraint
        self.unroll = unroll
        self.equilibration_params = equilibration_params
        self.setup()

    def setup(self) -> None:
        """Setup the projection layer."""
        self.constraints = [
            c
            for c in (self.eq_constraint, self.box_constraint, self.ineq_constraint)
            if c
        ]
        n_constraints = len(self.constraints)
        assert n_constraints > 0, "At least one constraint must be provided."
        self.n_constraints = n_constraints
        self.dim = self.constraints[0].dim
        # Costraints need to be parsed
        if self.ineq_constraint is not None or self.n_constraints > 1:
            if self.ineq_constraint is not None:
                self.dim_lifted = self.dim + self.ineq_constraint.n_constraints
            else:
                self.dim_lifted = self.dim
            parser = ConstraintParser(
                eq_constraint=self.eq_constraint,
                ineq_constraint=self.ineq_constraint,
                box_constraint=self.box_constraint,
            )
            self.lifted_eq_constraint, self.lifted_box_constraint = parser.parse(
                method=None
            )
            # Only equilibrate when we have a single A
            if (
                not self.lifted_eq_constraint.var_A
                and self.lifted_eq_constraint.A.shape[0] == 1
            ):
                scaled_A, d_r, d_c = ruiz_equilibration(
                    self.lifted_eq_constraint.A[0], self.equilibration_params
                )
                self.d_r = d_r.reshape(1, -1, 1)
                self.d_c = d_c.reshape(1, -1, 1)
                # Update A in lifted equality and setup projection
                self.lifted_eq_constraint.A = scaled_A.reshape(
                    1,
                    self.lifted_eq_constraint.A.shape[1],
                    self.lifted_eq_constraint.A.shape[2],
                )
                self.lifted_eq_constraint.method = "pinv"
                self.lifted_eq_constraint.setup()
                # Scale the equality RHS
                self.lifted_eq_constraint.b = self.lifted_eq_constraint.b * self.d_r
                # Scale the lifted box constraints
                self.lifted_box_constraint.upper_bound = (
                    self.lifted_box_constraint.upper_bound
                    / self.d_c[:, self.lifted_box_constraint.mask, :]
                )
                self.lifted_box_constraint.lower_bound = (
                    self.lifted_box_constraint.lower_bound
                    / self.d_c[:, self.lifted_box_constraint.mask, :]
                )
            else:
                # TODO: Think if it makes sense to do equilibration for variable A
                n_ineq = (
                    self.ineq_constraint.n_constraints
                    if self.ineq_constraint is not None
                    else 0
                )
                self.d_r = jnp.ones((1, self.eq_constraint.n_constraints + n_ineq, 1))
                self.d_c = jnp.ones((1, self.dim_lifted, 1))
                self.lifted_eq_constraint.method = "pinv"
                self.lifted_eq_constraint.setup()

            self.step_iteration, self.step_final = build_iteration_step(
                self.lifted_eq_constraint,
                self.lifted_box_constraint,
                self.dim,
                self.d_c[:, : self.dim, :],
            )
            # Equality constraint has variable A
            if self.lifted_eq_constraint.var_A:
                self._project = jax.jit(
                    partial(
                        _project_general,
                        self.step_iteration,
                        self.step_final,
                        self.dim_lifted,
                        self.d_r,
                        self.d_c,
                    ),
                    static_argnames=["n_iter"],
                )
            # Equality constraint has variable b
            elif self.lifted_eq_constraint.var_b:
                if self.unroll:
                    self._project = jax.jit(
                        partial(
                            _project_general,
                            self.step_iteration,
                            self.step_final,
                            self.dim_lifted,
                            self.d_r,
                            self.d_c,
                        ),
                        static_argnames=["n_iter"],
                    )
                else:
                    self._project = jax.jit(
                        partial(
                            _project_general_custom,
                            self.step_iteration,
                            self.step_final,
                            self.dim_lifted,
                            self.d_r,
                            self.d_c,
                        ),
                        static_argnames=["n_iter", "n_iter_bwd", "fpi"],
                    )
            # Equality constraint does not vary
            else:
                if self.unroll:
                    self._project = jax.jit(
                        partial(
                            _project_general,
                            self.step_iteration,
                            self.step_final,
                            self.dim_lifted,
                            self.d_r,
                            self.d_c,
                        ),
                        static_argnames=["n_iter"],
                    )
                else:
                    self._project = jax.jit(
                        partial(
                            _project_general_custom,
                            self.step_iteration,
                            self.step_final,
                            self.dim_lifted,
                            self.d_r,
                            self.d_c,
                        ),
                        static_argnames=["n_iter", "n_iter_bwd", "fpi"],
                    )

        else:
            self.single_constraint = self.constraints[0]
            self._project = jax.jit(self._project_single)

        # jit correctly the call method
        self.call = self._project

    def cv(self, inp: ProjectionInstance) -> jnp.ndarray:
        """Compute the constraint violation.

        Args:
            x (jnp.ndarray): Point to be evaluated.
                Shape (batch_size, dimension, 1).

        Returns:
            jnp.ndarray: Constraint violation for each point in the batch.
        """
        inp = inp.update(x=inp.x.reshape(inp.x.shape[0], inp.x.shape[1], 1))
        cv = jnp.zeros((inp.x.shape[0], 1, 1))
        for c in self.constraints:
            if c is not None:
                cv = jnp.maximum(cv, c.cv(inp))

        return cv

    def call_and_check(
        self,
        sigma=1.0,
        omega=1.7,
        check_every=10,
        tol=1e-3,
        max_iter=100,
        reduction="max",
    ) -> Callable:
        """Returns a function that projects input and checks constraint violation.

        Args:
            check_every (int): Frequency of checking constraint violation.
            tol (float): Tolerance for constraint violation.
            max_iter (int): Maximum number of iterations for checking.
            reduction (str): Method to reduce constraint violations among a batch.
            Valid options are: "max" meaning that maximum cv is less that tol;
            "mean" meaning that mean cv is less than tol;
            or a number in [0,1] meaning the percentage of instances
            with cv less than tol.

        Returns:
            Callable: Takes as input the points to be projected and any specifications for
            the constraints (e.g., the value of b for variable b equality constraints.).
            Returns an approximately project and a flag showing whether the termination
            condition was satisfied.
        """

        @jax.jit
        def check(inp: ProjectionInstance) -> bool:
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

        def project_and_check(inp: ProjectionInstance) -> tuple[jnp.ndarray, bool, int]:
            y0 = self.get_init(inp)
            # Executed iterations
            iter_exec = 0
            terminate = False
            # Call the projection function with all given arguments.
            while not (terminate or iter_exec >= max_iter):
                xproj, y = self.call(
                    y0,
                    inp,
                    sigma=sigma,
                    omega=omega,
                    n_iter=check_every,
                    n_iter_bwd=0,  # only used when backproping
                    fpi=False,  # only used when backproping
                )
                y0 = y0.update(x=y)
                iter_exec += check_every
                terminate = check(inp.update(x=xproj))

            return xproj.reshape(inp.x.shape), terminate, iter_exec

        return project_and_check

    def _project_single(self, inp: ProjectionInstance, _: int = 0) -> jnp.ndarray:
        """Project a batch of points with single constraint.

        Args:
            x (jnp.ndarray): Point to be projected.
                Shape (batch_size, dimension, 1).
            _ (int): Unused argument for compatibility.

        Returns:
            jnp.ndarray: The projected point for each point in the batch.
        """
        if (self.eq_constraint is not None) and (self.eq_constraint.var_A):
            Apinv = jnp.linalg.pinv(inp.eq.A)
            inp = inp.update(eq=inp.eq.update(Apinv=Apinv))

        return self.single_constraint.project(
            inp.update(x=inp.x.reshape((inp.x.shape[0], inp.x.shape[1], 1)))
        ).reshape(inp.x.shape)

    def get_init(self, inp: ProjectionInstance) -> ProjectionInstance:
        """Returns a zero initial value for the governing sequence.

        Args:
            inp (ProjectionInstance): Point to be projected data.

        Returns:
            ProjectionInstance: Initial value for the governing sequence.
        """
        inp = self.preprocess(inp)
        return inp.update(x=jnp.zeros((inp.x.shape[0], self.dim_lifted, 1)))

    def preprocess(self, inp: ProjectionInstance) -> ProjectionInstance:
        """Preprocess inputs to ease the projection process.

        For example, adds zeros to the RHS of changing equality
        constraints.

        Args:
            inp (ProjectionInstance): Point to be projected data.

        Returns:
            ProjectionInstance: Appropriately preprocessed inputs.
        """
        if self.eq_constraint is not None:
            if self.eq_constraint.var_A:
                parser = ConstraintParser(
                    eq_constraint=EqualityConstraint(inp.eq.A, inp.eq.b, method="pinv"),
                    ineq_constraint=self.ineq_constraint,
                    box_constraint=self.box_constraint,
                )
                lifted_eq_constraint, _ = parser.parse(method="pinv")
                inp = inp.update(
                    eq=inp.eq.update(
                        A=lifted_eq_constraint.A, Apinv=lifted_eq_constraint.Apinv
                    )
                )

            if self.eq_constraint.var_b:
                b_lifted = (
                    jnp.concatenate(
                        [
                            inp.eq.b,
                            jnp.zeros(
                                shape=(inp.eq.b.shape[0], self.dim_lifted - self.dim, 1)
                            ),
                        ],
                        axis=1,
                    )
                    * self.d_r
                )
                inp = inp.update(eq=inp.eq.update(b=b_lifted))

        return inp


# Project general
def _project_general(
    step_iteration: Callable[
        [ProjectionInstance, jnp.ndarray, float, float], ProjectionInstance
    ],
    step_final: Callable[[ProjectionInstance], jnp.ndarray],
    dim_lifted: int,
    d_r: jnp.ndarray,
    d_c: jnp.ndarray,
    y0: ProjectionInstance,
    inp: ProjectionInstance,
    sigma: float = 1.0,
    omega: float = 1.7,
    n_iter: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project a batch of points using Douglas-Rachford.

    Args:
        step_iteration (callable): Function for the iteration step.
        step_final (callable): Function for the final step.
        dim_lifted (int): Dimension of the lifted space.
        d_r (jnp.ndarray): Scaling factor for the rows.
        d_c (jnp.ndarray): Scaling factor for the columns.
        y0 (jnp.ndarray): Initial value for the governing sequence.
            Shape (batch_size, dim_lifted, 1).
        x (jnp.ndarray): Point to be projected.
            Shape (batch_size, dimension, 1).
        sigma (float): ADMM parameter.
        omega (float): ADMM parameter.
        n_iter (int): Number of iterations to run.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: First output is the projected
            point, and second output is the value of the governing sequence.
    """
    y = y0
    y, _ = jax.lax.scan(
        lambda y, _: (
            step_iteration(
                y, inp.x.reshape((inp.x.shape[0], inp.x.shape[1], 1)), sigma, omega
            ),
            None,
        ),
        y,
        None,
        length=n_iter,
    )
    y_aux = y.x
    # Unscale and reshape the output
    y = (step_final(y)[:, : inp.x.shape[1], :] * d_c[:, : inp.x.shape[1], :]).reshape(
        inp.x.shape
    )
    return y, y_aux


@partial(jax.custom_vjp, nondiff_argnums=[0, 1, 2, 10, 11, 12])
def _project_general_custom(
    step_iteration: Callable[
        [ProjectionInstance, jnp.ndarray, float, float], ProjectionInstance
    ],
    step_final: Callable[[ProjectionInstance], jnp.ndarray],
    dim_lifted: int,
    d_r: jnp.ndarray,
    d_c: jnp.ndarray,
    y0: ProjectionInstance,
    inp: ProjectionInstance,
    sigma: float = 1.0,
    omega: float = 1.7,
    n_iter: int = 0,
    n_iter_bwd: int = 5,
    fpi: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Auxiliary function to define custom vjp."""
    return _project_general(
        step_iteration,
        step_final,
        dim_lifted,
        d_r,
        d_c,
        y0,
        inp,
        sigma,
        omega,
        n_iter,
    )


def _project_general_fwd(
    step_iteration: Callable[[jnp.ndarray, jnp.ndarray, float, float], jnp.ndarray],
    step_final: Callable[[jnp.ndarray], jnp.ndarray],
    dim_lifted: int,
    d_r: jnp.ndarray,
    d_c: jnp.ndarray,
    y0: ProjectionInstance,
    inp: ProjectionInstance,
    sigma: float = 1.0,
    omega: float = 1.7,
    n_iter: int = 0,
    n_iter_bwd: int = 5,
    fpi: bool = False,
):
    """Forward pass for custom vjp."""
    y, y_aux = _project_general_custom(
        step_iteration,
        step_final,
        dim_lifted,
        d_r,
        d_c,
        y0,
        inp,
        sigma,
        omega,
        n_iter,
        n_iter_bwd,
        fpi,
    )
    return (y, y_aux), (
        y0.update(x=y_aux),
        inp.update(x=inp.x.reshape((inp.x.shape[0], inp.x.shape[1], 1))),
        d_r,
        d_c,
        sigma,
        omega,
    )


def _project_general_bwd(
    step_iteration: Callable[
        [ProjectionInstance, jnp.ndarray, float, float], ProjectionInstance
    ],
    step_final: Callable[[ProjectionInstance], jnp.ndarray],
    dim_lifted: int,
    n_iter: int,
    n_iter_bwd: int,
    fpi: bool,
    res: tuple,
    g: jnp.ndarray,
) -> tuple:
    """Backward pass for custom vjp.

    This function computes the vjp for the projection using the
    implicit function theorem.
    Note that, the arguments are:
    (i) any arguments for the
    forward that are not jnp.ndarray;
    (ii) res: tuple with auxiliary data from the forward pass;
    (iii) g: incoming cotangents.
    The function returns a tuple where each element corresponds
    to a jnp.ndarray from the input.

    Args:
        step_iteration (callable): Function for the iteration step.
        step_final (callable): Function for the final step.
        dim_lifted (int): Dimension of the lifted space.
        n_iter (int): Number of iterations to run.
        n_iter_bwd (int): Number of iterations for backward pass.
        fpi (bool): Whether to use fixed-point iteration.
        res (tuple): Auxiliary data from the forward pass.
        g (tuple): Incoming cotangents.

    Returns:
        tuple: The computed cotangent for the projection.
    """
    aux_proj, xproj, d_r, d_c, sigma, omega = res
    _, iteration_vjp = jax.vjp(
        lambda xx: step_iteration(aux_proj.update(x=xx), xproj.x, sigma, omega).x,
        aux_proj.x,
    )
    _, iteration_vjp2 = jax.vjp(
        lambda xx: step_iteration(aux_proj, xx, sigma, omega).x, xproj.x
    )
    _, equality_vjp = jax.vjp(lambda xx: step_final(aux_proj.update(x=xx)), aux_proj.x)

    # Rescale the gradient
    g_scaled = (
        g[0].reshape(g[0].shape[0], g[0].shape[1], 1) * d_c[:, : xproj.x.shape[1], :]
    )

    # Compute VJP of cotangent with projection before auxiliary
    gg = equality_vjp(
        jnp.concatenate(
            [
                g_scaled,
                jnp.zeros((g[0].shape[0], dim_lifted - g[0].shape[1], 1)),
            ],
            axis=1,
        )
    )[0]
    # Run iteration
    if fpi:
        vjp_iter = jnp.zeros((g[0].shape[0], dim_lifted, 1))
        vjp_iter, _ = jax.lax.scan(
            lambda vjp_iter, _: (iteration_vjp(vjp_iter)[0] + gg, None),
            vjp_iter,
            None,
            length=n_iter_bwd,
        )
    else:

        def Aop(xx):
            return xx - iteration_vjp(xx)[0]

        vjp_iter = jax.scipy.sparse.linalg.bicgstab(Aop, gg, maxiter=n_iter_bwd)[0]
    thevjp = iteration_vjp2(vjp_iter)[0].reshape((g[0].shape[0], g[0].shape[1]))
    return (None, None, None, ProjectionInstance(x=thevjp), None, None)


_project_general_custom.defvjp(_project_general_fwd, _project_general_bwd)
