"""Flax implementation of the projection layer."""

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


# TODO: When using var_A, we should rename the Apinv argument that is passed around
# to something that generically represents a factorization of A.
class Project:
    """Projection layer implemented via iterative projections."""

    eq_constraint: EqualityConstraint = None
    ineq_constraint: AffineInequalityConstraint = None
    box_constraint: BoxConstraint = None

    def __init__(
        self,
        eq_constraint: EqualityConstraint = None,
        ineq_constraint: AffineInequalityConstraint = None,
        box_constraint: BoxConstraint = None,
    ):
        """Initialize projection layer.

        Args:
            eq_constraint (EqualityConstraint): Equality constraint.
            ineq_constraint (AffineInequalityConstraint): Inequality
                constraint.
            box_constraint (BoxConstraint): Box constraint.
        """
        self.eq_constraint = eq_constraint
        self.ineq_constraint = ineq_constraint
        self.box_constraint = box_constraint
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
        if self.ineq_constraint is not None or self.n_constraints > 1:
            self.dim_lifted = self.dim + self.ineq_constraint.n_constraints
            parser = ConstraintParser(
                eq_constraint=self.eq_constraint,
                ineq_constraint=self.ineq_constraint,
                box_constraint=self.box_constraint,
            )
            self.lifted_eq_constraint, self.lifted_ineq_constraint = parser.parse()
            if self.lifted_eq_constraint.var_A:
                self.step_iteration, self.step_final = build_iteration_step_vAb(
                    self.lifted_eq_constraint, self.lifted_ineq_constraint, self.dim
                )
                self._project = jax.jit(
                    self._project_general_vAb, static_argnames=["n_iter"]
                )
            elif self.lifted_eq_constraint.var_b:
                self.step_iteration, self.step_final = build_iteration_step_vb(
                    self.lifted_eq_constraint, self.lifted_ineq_constraint, self.dim
                )
                self._project = jax.jit(
                    self._project_general_vb, static_argnames=["n_iter"]
                )
            else:
                self.step_iteration, self.step_final = build_iteration_step(
                    self.lifted_eq_constraint, self.lifted_ineq_constraint, self.dim
                )
                self._project = jax.jit(
                    self._project_general, static_argnames=["n_iter"]
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

    def _project_general(
        self, x: jnp.ndarray, interpolation_value: float = 0, n_iter: int = 0
    ) -> jnp.ndarray:
        y = jnp.zeros(shape=(x.shape[0], self.dim_lifted, 1))
        y, _ = jax.lax.scan(
            lambda y, _: (
                self.step_iteration(y, x.reshape((x.shape[0], x.shape[1], 1))),
                None,
            ),
            y,
            None,
            length=n_iter,
        )
        y = self.step_final(y).reshape(x.shape)
        return interpolation_value * x + (1 - interpolation_value) * y

    def _project_general_vb(
        self,
        x: jnp.ndarray,
        b: jnp.ndarray,
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
        y = jnp.zeros(shape=(x.shape[0], self.dim_lifted, 1))
        y, _ = jax.lax.scan(
            lambda y, _: (
                self.step_iteration(
                    y, x.reshape((x.shape[0], x.shape[1], 1)), b_lifted
                ),
                None,
            ),
            y,
            None,
            length=n_iter,
        )
        y = self.step_final(y, b_lifted).reshape(x.shape)
        return interpolation_value * x + (1 - interpolation_value) * y

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

    def __call__(
        self, x: jnp.ndarray, interpolation_value: float = 0, n_iter: int = 0
    ) -> jnp.ndarray:
        """Project the input to the feasible region.

        Args:
            x (jnp.ndarray): Input tensor.
            interpolation_value (float, optional):
                Interpolation value between the input and the projection.
            n_iter (int, optional): Number of iterations for the projection.
        """
        # Make sure x is of shape (batch_size, n_dims, 1)
        y = self._project(x.reshape((x.shape[0], x.shape[1], 1)), n_iter).reshape(
            x.shape
        )
        return interpolation_value * x + (1 - interpolation_value) * y
