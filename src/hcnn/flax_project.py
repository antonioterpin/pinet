"""Flax implementation of the projection layer."""

from flax import linen as nn
from jax import numpy as jnp

from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.constraints.box import BoxConstraint
from hcnn.constraints.constraint_parser import ConstraintParser
from hcnn.solver.admm import build_iteration_step


class Project(nn.Module):
    """Projection layer implemented via iterative projections."""

    eq_constraint: EqualityConstraint = None
    ineq_constraint: AffineInequalityConstraint = None
    box_constraint: BoxConstraint = None

    def setup(self):
        """Setup the projection layer."""
        constraints = [
            c
            for c in (self.eq_constraint, self.box_constraint, self.ineq_constraint)
            if c
        ]
        n_constraints = len(constraints)
        assert n_constraints > 0, "At least one constraint must be provided."
        self.dim = constraints[0].dim
        if self.ineq_constraint is not None or n_constraints > 1:
            self.dim_lifted = self.dim + self.ineq_constraint.shape[-1]
            parser = ConstraintParser(
                eq_constraint=self.eq_constraint,
                ineq_constraint=self.ineq_constraint,
                box_constraint=self.box_constraint,
            )
            self.lifted_eq_constraint, self.lifted_ineq_constraint = parser.parse()
            self.step_iteration, self.step_final = build_iteration_step(
                self.lifted_eq_constraint, self.lifted_ineq_constraint, self.dim
            )
            self._project = self._project_general
        else:
            self.single_constraint = constraints[0]
            self._project = self._project_single

    def _project_general(self, x: jnp.ndarray, n_iter: int) -> jnp.ndarray:
        y = jnp.zeros(shape=(x.shape[0], self.dim_lifted, 1))
        for _ in range(n_iter):
            y = self.step_iteration(y, x)
        y = self.step_final(y)
        return y

    def _project_single(self, x: jnp.ndarray, _: int) -> jnp.ndarray:
        y = jnp.expand_dims(x, axis=2)
        y = self.single_constraint.project(y)
        return y.reshape(x.shape)

    @nn.compact
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
        y = self._project(x, n_iter)
        return interpolation_value * x + (1 - interpolation_value) * y
