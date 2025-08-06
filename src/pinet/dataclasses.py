"""This file contains dataclasses used to encapsulate inputs for the Pinet layer."""

from dataclasses import dataclass, replace
from typing import Optional

import jax
import jax.numpy as jnp


# Inputs dataclasses
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EqualityConstraintsSpecification:
    """Dataclass representing inputs used in forming equality constraints.

    Attributes:
        b (Optional[jnp.ndarray]): Vector representing the RHS of the equality constraint.
            Shape (batch_size, n_constraints, 1)
        A (Optional[jnp.ndarray]): Matrix representing the LHS of the equality constraint.
            Shape (batch_size, n_constraints, dimension).
        Apinv (Optional[jnp.ndarray]): The pseudoinverse of the matrix A.
            Shape (batch_size, dimension, n_constraints).
    """

    b: Optional[jnp.ndarray] = None
    A: Optional[jnp.ndarray] = None
    Apinv: Optional[jnp.ndarray] = None

    def update(self, **kwargs):
        """Update some attribute by keyword."""
        return replace(self, **kwargs)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AffineInequalityConstraintSpecification:
    """Dataclass representing affine inequality constraints.

    Attributes:
        C (jnp.ndarray): Coefficient matrix for the inequality constraints.
            Shape (batch_size, n_constraints, dimension).
        lb (jnp.ndarray): Lower bound vector for the inequality constraints.
            Shape (batch_size, n_constraints, 1).
        ub (jnp.ndarray): Upper bound vector for the inequality constraints.
            Shape (batch_size, n_constraints, 1).
    """

    C: jnp.ndarray
    lb: Optional[jnp.ndarray] = None
    ub: Optional[jnp.ndarray] = None

    def update(self, **kwargs):
        """Update some attribute by keyword."""
        return replace(self, **kwargs)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BoxConstraintSpecification:
    """Dataclass representing box constraints.

    Attributes:
        lb (Optional[jnp.ndarray]): Lower bounds for the box constraints.
            Shape (batch_size, dimension, 1).
        ub (Optional[jnp.ndarray]): Upper bounds for the box constraints.
            Shape (batch_size, dimension, 1).
    """

    lb: Optional[jnp.ndarray] = None
    ub: Optional[jnp.ndarray] = None

    def update(self, **kwargs):
        """Update some attribute by keyword."""
        return replace(self, **kwargs)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ProjectionInstance:
    """A dataclass for encapsulating model input parameters.

    Attributes:
        x (jnp.ndarray): The point to be projected.
            Shape (batch_size, dimension, 1)
        eq (Optional[EqualityConstraintsSpecification]):
            Specification of the equality constraints, if any.
        ineq (Optional[AffineInequalityConstraintSpecification]):
            Specification of the affine inequality constraints, if any.
        box (Optional[BoxConstraintSpecification]):
            Specification of the box constraints, if any.
    """

    x: jnp.ndarray
    eq: Optional[EqualityConstraintsSpecification] = None
    ineq: Optional[AffineInequalityConstraintSpecification] = None
    box: Optional[BoxConstraintSpecification] = None

    def update(self, **kwargs):
        """Update some attribute by keyword."""
        return replace(self, **kwargs)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EquilibrationParams:
    """A dataclass for encapsulating the equilibration parameters.

    Attributes:
        max_iter (int): Maximum number of iterations for the equilibration.
        tol (float): Tolerance for convergence of the equilibration.
        ord (float): Order of the norm used for convergence check.
        col_scaling (bool): Whether to apply column scaling.
        update_mode (str): Update mode for the equilibration.
            Available options are:
                - "Jacobi" means compute both row and column norms and update.
                - "Gauss" means compute row, update, compute column, update.
        safeguard (bool): Check if the condition number of A has decreased.
    """

    max_iter: int = 0
    tol: float = 1e-3
    ord: float = 2.0
    col_scaling: bool = False
    update_mode: str = "Gauss"
    safeguard: bool = False

    def update(self, **kwargs):
        """Update some attribute by keyword."""
        if "update_mode" in kwargs:
            assert kwargs["update_mode"] in [
                "Gauss",
                "Jacobi",
            ], 'update_mode must be either "Gauss" or "Jacobi"'

        return replace(self, **kwargs)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ProjectionParams:
    """A dataclass for encapsulating projection parameters.

    Attributes:
        sigma (float): Proximal step size for the projection.
        omega (float): Relaxation parameter for the projection.
        max_iter (int): Maximum number of iterations for the projection.
        tol (float): Tolerance for convergence of the projection.
        reduction (float|str): Type of reduction to apply for the tolerance check.
            Available options are:
                - "max": Maximum value of the residuals.
                - "mean": Mean value of the residuals.
                - float value: Percentage of the residuals below tol.
    """

    def update(self, **kwargs):
        """Update some attribute by keyword."""
        return replace(self, **kwargs)
