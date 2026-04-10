"""Projection layers using other approaches."""

from collections.abc import Callable
from typing import cast

import cvxpy as cp
import jax
import jax.numpy as jnp
import jaxopt
from cvxpylayers.jax import CvxpyLayer

jax.config.update("jax_enable_x64", True)


def get_jaxopt_projection(
    a_dyn: jnp.ndarray,
    constr_matrix: jnp.ndarray,
    d: jnp.ndarray,
    dim: int,
    tol: float = 1e-3,
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Compute a batched projection function for polyhedral constraints using JAXopt.

    This function creates a projection operator using the jaxopt.OSQP solver.
    The projection is formulated as the quadratic program:
    minimize   (1/2) * ||x - xx||^2
    subject to a_dyn x = b
               constr_matrix x <= d,
    where the quadratic term is given by the identity matrix of size `dim`.

    The resulting function is JIT-compiled and vectorized.

    Args:
        a_dyn: Coefficient matrix for equality constraints.
        constr_matrix: Coefficient matrix for inequality constraints.
        d: Right-hand side vector for inequality constraints.
        dim: Dimension of the variable x.
        tol: Tolerance for the solver. Defaults to 1e-3.

    Returns:
        Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        A JIT-compiled and vectorized function
        that takes a batch of input vectors (shape: (batch_size, dim))
        and returns their corresponding projections.
    """
    qp = jaxopt.OSQP(tol=tol)
    q_mat = jnp.eye(dim)
    jaxopt_proj = jax.jit(
        jax.vmap(
            lambda xx, bb: (
                qp.run(
                    params_obj=(q_mat, -xx),
                    params_eq=(a_dyn, bb[:, 0]),
                    params_ineq=(constr_matrix, d),
                ).params.primal
            ),
            in_axes=[0, 0],
        )
    )

    return jaxopt_proj


def get_cvxpy_projection(
    a_dyn: jnp.ndarray,
    constr_matrix: jnp.ndarray,
    d: jnp.ndarray,
    dim: int,
) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray]]:
    """Constructs and returns a CVXPY-based projection layer callable.

    The projection is formulated as a quadratic minimization problem that minimizes
    the squared distance between the projection variable and an input point xproj, subject
    to the constraints:
        a_dyn @ y = b   (equality constraints)
        constr_matrix @ y <= d  (inequality constraints)

    Args:
        a_dyn: Coefficient matrix for equality constraints.
        constr_matrix: Coefficient matrix for inequality constraints.
        d: Right-hand side vector for inequality constraints.
        dim: Dimension of the variable x.

    Returns:
        Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray]]:
        A callable CVXPY layer that takes two parameters:
        an input vector (xproj) to be projected and a corresponding vector b for
        the equality constraints.
        The callable returns the projected vector as a jnp.ndarray.
    """
    n_eq = a_dyn.shape[0]
    ycvxpy = cp.Variable(dim)
    xproj = cp.Parameter(dim)
    b = cp.Parameter(n_eq)
    constraints = cast(
        list[cp.Constraint],
        [
            a_dyn @ ycvxpy == b,
            constr_matrix @ ycvxpy <= d,
        ],
    )
    objective = cp.Minimize(cp.sum_squares(ycvxpy - xproj))
    problem_cvxpy = cp.Problem(objective=objective, constraints=constraints)
    assert problem_cvxpy.is_dpp()

    cvxpylayer = CvxpyLayer(
        problem_cvxpy,
        parameters=[xproj, b],
        variables=[ycvxpy],
    )

    return cvxpylayer
