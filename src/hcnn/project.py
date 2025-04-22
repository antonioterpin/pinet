"""Flax implementation of the projection layer."""

from functools import partial

import jax
from jax import numpy as jnp

from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.constraints.box import BoxConstraint
from hcnn.constraints.constraint_parser import ConstraintParser
from hcnn.solver.admm import (
    build_iteration_step,
    build_iteration_step_vAb,
    build_iteration_step_vb,
)


# TODO: During inference, it would be nice to
#       implement some constraint violation checking to reduce runtime.
# TODO: Make the output of project more consistent.
#       For single constraints the output is an array.
#       For parsed/multiple constraints the output is a tuple.
# TODO: Remove the __call__ method, and maybe rename the currently
#   used `call` method to something else, e.g., project.
# TODO: Remove the interpolation value. This should be done on the NN layer.
# TODO: When using var_A, we should rename the Apinv argument that is passed around
# to something that generically represents a factorization of A.
# TODO: Break down the iteration vjps, so they
#   do not have to be computed every time we back prop.
#   It does not seem to be taking very long so
#   maybe we can leave this for a later stage.
class Project:
    """Projection layer implemented via iterative projections."""

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
    ):
        """Initialize projection layer.

        Args:
            eq_constraint (EqualityConstraint): Equality constraint.
            ineq_constraint (AffineInequalityConstraint): Inequality
                constraint.
            box_constraint (BoxConstraint): Box constraint.
            sigma (float): ADMM scaling parameter.
            omega (float): ADMM relaxation parameter.
            unroll (bool): Use loop unrolling for backpropagation.
        """
        self.eq_constraint = eq_constraint
        self.ineq_constraint = ineq_constraint
        self.box_constraint = box_constraint
        self.unroll = unroll
        self.setup()

    def setup(self):
        """Setup the projection layer."""
        constraints = [
            c
            for c in (self.eq_constraint, self.box_constraint, self.ineq_constraint)
            if c
        ]
        n_constraints = len(constraints)
        assert n_constraints > 0, "At least one constraint must be provided."
        self.n_constraints = n_constraints
        self.dim = constraints[0].dim
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
            # TODO: Change lifted_ineq to lifted_box
            self.lifted_eq_constraint, self.lifted_ineq_constraint = parser.parse()

            # Equality constraint has variable A
            if self.lifted_eq_constraint.var_A:
                self.step_iteration, self.step_final = build_iteration_step_vAb(
                    self.lifted_eq_constraint,
                    self.lifted_ineq_constraint,
                    self.dim,
                )
                self._project = jax.jit(
                    self._project_general_vAb, static_argnames=["n_iter"]
                )
            # Equality constraint has variable b
            elif self.lifted_eq_constraint.var_b:
                self.step_iteration, self.step_final = build_iteration_step_vb(
                    self.lifted_eq_constraint,
                    self.lifted_ineq_constraint,
                    self.dim,
                )
                if self.unroll:
                    self._project = jax.jit(
                        partial(
                            _project_general_vb,
                            self.step_iteration,
                            self.step_final,
                            self.dim_lifted,
                        ), static_argnames=["n_iter"],
                    )
                else:
                    self._project = jax.jit(
                        partial(
                            _project_general_vb_custom,
                            self.step_iteration,
                            self.step_final,
                            self.dim_lifted,
                        ),
                        static_argnames=["n_iter", "n_iter_bwd", "fpi"],
                    )
            # Equality constraint does not vary
            else:
                self.step_iteration, self.step_final = build_iteration_step(
                    self.lifted_eq_constraint,
                    self.lifted_ineq_constraint,
                    self.dim,
                )
                if self.unroll:
                    self._project = jax.jit(
                        partial(
                            _project_general,
                            self.step_iteration,
                            self.step_final,
                            self.dim_lifted,
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
                        ),
                        static_argnames=["n_iter", "n_iter_bwd", "fpi"],
                    )

        else:
            self.single_constraint = constraints[0]
            if self.eq_constraint is not None:
                if self.eq_constraint.var_A:
                    self._project = jax.jit(self._project_single_vAb)
                elif self.eq_constraint.var_b:
                    self._project = jax.jit(self._project_single_vb)
                else:
                    self._project = jax.jit(self._project_single)
            else:
                self._project = jax.jit(self._project_single)

        # jit correctly the call method
        self.call = self._project

    def _project_general_vAb(
        self,
        x: jnp.ndarray,
        b: jnp.ndarray,
        A: jnp.ndarray,
        interpolation_value: float = 0,
        n_iter: int = 0,
    ) -> jnp.ndarray:
        # First write in lifted formulation
        b_lifted = jnp.concatenate(
            [
                b,
                jnp.zeros(shape=(b.shape[0], self.ineq_constraint.n_constraints, 1)),
            ],
            axis=1,
        )
        if self.lifted_eq_constraint.method == "pinv":
            if self.ineq_constraint is not None or self.n_constraints > 1:
                parser = ConstraintParser(
                    eq_constraint=EqualityConstraint(A, b, method="pinv"),
                    ineq_constraint=self.ineq_constraint,
                    box_constraint=self.box_constraint,
                )
                lifted_eq_constraint, _ = parser.parse(method="pinv")
        else:
            assert False

        y = jnp.zeros(shape=(x.shape[0], self.dim_lifted, 1))
        y, _ = jax.lax.scan(
            lambda y, _: (
                self.step_iteration(
                    y,
                    x.reshape((x.shape[0], x.shape[1], 1)),
                    b_lifted,
                    lifted_eq_constraint.A,
                    lifted_eq_constraint.Apinv,
                ),
                None,
            ),
            y,
            None,
            length=n_iter,
        )
        y = self.step_final(
            y, b_lifted, lifted_eq_constraint.A, lifted_eq_constraint.Apinv
        ).reshape(x.shape)
        return interpolation_value * x + (1 - interpolation_value) * y

    def _project_single(
        self, x: jnp.ndarray, interpolation_value: float = 0, _: int = 0
    ) -> jnp.ndarray:
        y = self.single_constraint.project(
            x.reshape((x.shape[0], x.shape[1], 1))
        ).reshape(x.shape)
        return interpolation_value * x + (1 - interpolation_value) * y

    def _project_single_vb(
        self, x: jnp.ndarray, b: jnp.ndarray, interpolation_value: float = 0, _: int = 0
    ) -> jnp.ndarray:
        y = self.single_constraint.project(
            x.reshape((x.shape[0], x.shape[1], 1)), b
        ).reshape(x.shape)
        return interpolation_value * x + (1 - interpolation_value) * y

    def _project_single_vAb(
        self,
        x: jnp.ndarray,
        b: jnp.ndarray,
        A: jnp.ndarray,
        interpolation_value: float = 0,
        _: int = 0,
    ) -> jnp.ndarray:
        if self.eq_constraint.method == "pinv":
            Apinv = jnp.linalg.pinv(A)
        else:
            assert False
        y = self.single_constraint.project(
            x.reshape((x.shape[0], x.shape[1], 1)), b, A, Apinv
        ).reshape(x.shape)
        return interpolation_value * x + (1 - interpolation_value) * y



# Project general
def _project_general(
    step_iteration,
    step_final,
    dim_lifted: int,
    x: jnp.ndarray,
    interpolation_value: float = 0,
    sigma: float = 1.0,
    omega: float = 1.7,
    n_iter: int = 0,
) -> jnp.ndarray:
    y = jnp.zeros(shape=(x.shape[0], dim_lifted, 1))
    y, _ = jax.lax.scan(
        lambda y, _: (
            step_iteration(
                y, x.reshape((x.shape[0], x.shape[1], 1)), sigma, omega
            ),
            None,
        ),
        y,
        None,
        length=n_iter,
    )
    y_aux = y
    y = step_final(y)[:, : x.shape[1], :].reshape(x.shape)
    return interpolation_value * x + (1 - interpolation_value) * y, y_aux


@partial(jax.custom_vjp, nondiff_argnums=[0, 1, 2, 7, 8, 9])
def _project_general_custom(
    step_iteration,
    step_final,
    dim_lifted: int,
    x: jnp.ndarray,
    interpolation_value: float = 0,
    sigma: float = 1.0,
    omega: float = 1.7,
    n_iter: int = 0,
    n_iter_bwd: int = 5,
    fpi: bool = False,
):
    return _project_general(
        step_iteration, 
        step_final, dim_lifted, x, interpolation_value, sigma, omega, n_iter
    )


def _project_general_fwd(
    step_iteration,
    step_final,
    dim_lifted: int,
    x: jnp.ndarray,
    interpolation_value: float = 0,
    sigma: float = 1.0,
    omega: float = 1.7,
    n_iter: int = 0,
    n_iter_bwd: int = 5,
    fpi: bool = False,
):
    y, y_aux = _project_general_custom(
        step_iteration,
        step_final,
        dim_lifted,
        x,
        interpolation_value,
        sigma,
        omega,
        n_iter,
        n_iter_bwd,
        fpi,
    )
    return (y, y_aux), (y_aux, x.reshape((x.shape[0], x.shape[1], 1)), sigma, omega)


def _project_general_bwd(
    step_iteration, step_final, dim_lifted, n_iter, n_iter_bwd, fpi, res, g
):
    aux_proj, xproj, sigma, omega = res
    _, iteration_vjp = jax.vjp(
        lambda xx: step_iteration(xx, xproj, sigma, omega), aux_proj)
    _, iteration_vjp2 = jax.vjp(
        lambda xx: step_iteration(aux_proj, xx, sigma, omega), xproj)
    _, equality_vjp = jax.vjp(lambda xx: step_final(xx), aux_proj)

    # Compute VJP of cotangent with projection before auxiliary
    gg = equality_vjp(
        jnp.concatenate(
            [
                g[0].reshape(g[0].shape[0], g[0].shape[1], 1),
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
    return (thevjp, None, None, None)


_project_general_custom.defvjp(_project_general_fwd, _project_general_bwd)


# Project general variable b
def _project_general_vb(
    step_iteration,
    step_final,
    dim_lifted: int,
    x: jnp.ndarray,
    b: jnp.ndarray,
    interpolation_value: float = 0,
    sigma: float = 1.0,
    omega: float = 1.7,
    n_iter: int = 0,
) -> jnp.ndarray:
    dim = x.shape[1]
    # First write in lifted formulation
    b_lifted = jnp.concatenate(
        [
            b,
            jnp.zeros(shape=(b.shape[0], dim_lifted - dim, 1)),
        ],
        axis=1,
    )
    y = jnp.zeros(shape=(x.shape[0], dim_lifted, 1))
    y = y.at[:, :dim, 0].set(x)
    y, _ = jax.lax.scan(
        lambda y, _: (
            step_iteration(
                y, x.reshape((x.shape[0], x.shape[1], 1)), b_lifted, sigma, omega),
            None,
        ),
        y,
        None,
        length=n_iter,
    )
    y_aux = y
    y = step_final(y, b_lifted)[:, : x.shape[1], :].reshape(x.shape)
    return interpolation_value * x + (1 - interpolation_value) * y, y_aux


@partial(jax.custom_vjp, nondiff_argnums=[0, 1, 2, 8, 9, 10])
def _project_general_vb_custom(
    step_iteration,
    step_final,
    dim_lifted: int,
    x: jnp.ndarray,
    b: jnp.ndarray,
    interpolation_value: float = 0,
    sigma: float = 1.0,
    omega: float = 1.7,
    n_iter: int = 0,
    n_iter_bwd: int = 5,
    fpi: bool = False,
):
    return _project_general_vb(
        step_iteration, 
        step_final, 
        dim_lifted, 
        x, 
        b, 
        interpolation_value, 
        sigma, 
        omega, 
        n_iter
    )


def _project_general_vb_fwd(
    step_iteration,
    step_final,
    dim_lifted: int,
    x: jnp.ndarray,
    b: jnp.ndarray,
    interpolation_value: float = 0,
    sigma: float = 1.0,
    omega: float = 1.7,
    n_iter: int = 0,
    n_iter_bwd: int = 5,
    fpi: bool = False,
):
    dim = x.shape[1]
    # First write in lifted formulation
    b_lifted = jnp.concatenate(
        [
            b,
            jnp.zeros(shape=(b.shape[0], dim_lifted - dim, 1)),
        ],
        axis=1,
    )
    y, y_aux = _project_general_vb_custom(
        step_iteration,
        step_final,
        dim_lifted,
        x,
        b,
        interpolation_value,
        sigma,
        omega,
        n_iter,
        n_iter_bwd,
        fpi,
    )
    return (y, y_aux), (
        y_aux, x.reshape((x.shape[0], x.shape[1], 1)), b_lifted, sigma, omega)


def _project_general_vb_bwd(
    step_iteration, step_final, dim_lifted, n_iter, n_iter_bwd, fpi, res, g
):
    aux_proj, xproj, b_lifted, sigma, omega = res
    _, iteration_vjp = jax.vjp(
        lambda xx: step_iteration(xx, xproj, b_lifted, sigma, omega), aux_proj)
    _, iteration_vjp2 = jax.vjp(
        lambda xx: step_iteration(aux_proj, xx, b_lifted, sigma, omega), xproj
    )
    _, equality_vjp = jax.vjp(lambda xx: step_final(xx, b_lifted), aux_proj)

    # Compute VJP of cotangent with projection before auxiliary
    gg = equality_vjp(
        jnp.concatenate(
            [
                g[0].reshape(g[0].shape[0], g[0].shape[1], 1),
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
    return (thevjp, None, None, None, None)


_project_general_vb_custom.defvjp(_project_general_vb_fwd, _project_general_vb_bwd)

# An interesting take away from this iterative approach
# is that you need to get the active constraints right.
# Once you have the correct active constraints,
# then you can just compute the gradient
# using the VJPs.
# Even more interesting, we might be able to seriously bring
# down the size of the backpropagation iteration,
# by exploiting the fact that we only care about the active
# constraints.
# Concretely, we can ran the back iteration
# for a much smaller size of augment auxiliary variable.
##################
# Regardles of the previous (but leading there), I can
# implement a solution guessing, where I round the
# constraints that appear active.
