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
    a_mat: jax.Array,
    c_mat: jax.Array,
    d: jax.Array,
    dim: int,
    tol: float = 1e-3,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Compute a batched projection function for polyhedral constraints using JAXopt.

    This function creates a projection operator using the jaxopt.OSQP solver.
    The projection is formulated as the quadratic program:
    minimize   (1/2) * ||x - xx||^2
    subject to a_mat x = b
               c_mat x <= d,
    where the quadratic term is given by the identity matrix of size `dim`.

    The resulting function is JIT-compiled and vectorized.

    Args:
        a_mat: Coefficient matrix for equality constraints.
        c_mat: Coefficient matrix for inequality constraints.
        d: Right-hand side vector for inequality constraints.
        dim: Dimension of the variable x.
        tol: Tolerance for the solver. Defaults to 1e-3.

    Returns:
        Callable[[jax.Array, jax.Array], jax.Array]:
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
                    params_eq=(a_mat, bb[:, 0]),
                    params_ineq=(c_mat, d),
                ).params.primal
            ),
            in_axes=[0, 0],
        )
    )

    return jaxopt_proj


def get_cvxpy_projection(
    a_mat: jax.Array,
    c_mat: jax.Array,
    d: jax.Array,
    dim: int,
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, ...]]:
    """Constructs and returns a CVXPY-based projection layer callable.

    The projection is formulated as a quadratic minimization problem that minimizes
    the squared distance between the projection variable and an input point xproj, subject
    to the constraints:
        a_mat @ y = b   (equality constraints)
        c_mat @ y <= d  (inequality constraints)

    Args:
        a_mat: Coefficient matrix for equality constraints.
        c_mat: Coefficient matrix for inequality constraints.
        d: Right-hand side vector for inequality constraints.
        dim: Dimension of the variable x.

    Returns:
        Callable[[jax.Array, jax.Array], tuple[jax.Array, ...]]:
        A callable CVXPY layer that takes two parameters:
        an input vector (xproj) to be projected and a corresponding vector b for
        the equality constraints.
        The callable returns the projected vector as a jax.Array.
    """
    n_eq = a_mat.shape[0]
    ycvxpy = cp.Variable(dim)
    xproj = cp.Parameter(dim)
    b = cp.Parameter(n_eq)
    constraints = cast(
        list[cp.Constraint],
        [
            a_mat @ ycvxpy == b,
            c_mat @ ycvxpy <= d,
        ],
    )
    objective = cp.Minimize(cp.sum_squares(ycvxpy - xproj))
    problem_cvxpy = cp.Problem(objective=objective, constraints=constraints)
    # The cvxpylayer backend requires the problem to satisfy DPP rules.
    assert problem_cvxpy.is_dpp()

    cvxpylayer = CvxpyLayer(
        problem_cvxpy,
        parameters=[xproj, b],
        variables=[ycvxpy],
    )

    return cvxpylayer
