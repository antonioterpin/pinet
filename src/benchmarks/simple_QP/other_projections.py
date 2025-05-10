"""Projection layers using other approaches."""

from typing import Callable

import cvxpy as cp
import jax
import jax.numpy as jnp
import jaxopt
from cvxpylayers.jax import CvxpyLayer

from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.project import Project

jax.config.update("jax_enable_x64", True)


def get_jaxopt_projection(
    A: jnp.ndarray, C: jnp.ndarray, d: jnp.ndarray, dim: int, tol=1e-3
) -> Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    """Compute a batched projection function for polyhedral constraints using JAXopt.

    This function creates a projection operator using the jaxopt.OSQP solver.
    The projection is formulated as the quadratic program:
    minimize   (1/2) * ||x - xx||^2
    subject to A x = b
               C x <= d,
    where the quadratic term is given by the identity matrix of size `dim`.

    The resulting function is JIT-compiled and vectorized.

    Args:
        A (jnp.ndarray): Coefficient matrix for equality constraints.
        C (jnp.ndarray): Coefficient matrix for inequality constraints.
        d (jnp.ndarray): Right-hand side vector for inequality constraints.
        dim (int): Dimension of the variable x.
        tol (float, optional): Tolerance for the solver. Defaults to 1e-3.

    Returns:
        Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        A JIT-compiled and vectorized function
        that takes a batch of input vectors (shape: (batch_size, dim))
        and returns their corresponding projections.
    """
    qp = jaxopt.OSQP(tol=tol)
    Q = jnp.eye(dim)
    jaxopt_proj = jax.jit(
        jax.vmap(
            lambda xx, bb: qp.run(
                params_obj=(Q, -xx), params_eq=(A, bb[:, 0]), params_ineq=(C, d)
            ).params.primal,
            in_axes=[0, 0],
        )
    )

    return jaxopt_proj


def get_cvxpy_projection(
    A: jnp.ndarray,
    C: jnp.ndarray,
    d: jnp.ndarray,
    dim: int,
) -> Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray]]:
    """Constructs and returns a CVXPY-based projection layer callable.

    The projection is formulated as a quadratic minimization problem that minimizes
    the squared distance between the projection variable and an input point xproj, subject
    to the constraints:
        A @ y = b   (equality constraints)
        C @ y <= d  (inequality constraints)

    Args:
        A (jnp.ndarray): Coefficient matrix for equality constraints.
        C (jnp.ndarray): Coefficient matrix for inequality constraints..
        d (jnp.ndarray): Right-hand side vector for inequality constraints.
        dim (int): Dimension of the variable x.

    Returns:
        Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray]]:
        A callable CVXPY layer that takes two parameters:
        an input vector (xproj) to be projected and a corresponding vector b for
        the equality constraints.
        The callable returns the projected vector as a jnp.ndarray.
    """
    n_eq = A.shape[0]
    ycvxpy = cp.Variable(dim)
    xproj = cp.Parameter(dim)
    b = cp.Parameter(n_eq)
    constraints = [
        A @ ycvxpy == b,
        C @ ycvxpy <= d,
    ]
    objective = cp.Minimize(cp.sum_squares(ycvxpy - xproj))
    problem_cvxpy = cp.Problem(objective=objective, constraints=constraints)
    assert problem_cvxpy.is_dpp()

    cvxpylayer = CvxpyLayer(
        problem_cvxpy,
        parameters=[xproj, b],
        variables=[ycvxpy],
    )

    return cvxpylayer


if __name__ == "__main__":
    seed = 42
    batch_size = 1
    # Generate random feasible problem
    dim = 100
    n_eq = 40
    n_ineq = 50
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, 5 + batch_size)
    # Generate equality constraint LHS
    A = jax.random.normal(key[0], (1, n_eq, dim))
    # Generate inequality constraints LHS
    C = jax.random.normal(key[1], shape=(1, n_ineq, dim))
    b = jnp.zeros(shape=(batch_size, n_eq, 1))
    lb = jax.random.uniform(key[2], shape=(1, n_ineq, 1), minval=-1.0, maxval=0.0)
    delta = jax.random.uniform(key[3], shape=(1, n_ineq, 1), minval=0.1, maxval=1.0)
    ub = lb + delta

    # Compute RHS by solving feasibility problems
    for ii in range(batch_size):
        xfeas = cp.Variable(dim)
        bfeas = cp.Variable(n_eq)
        constraints = [
            A[0, :, :] @ xfeas == bfeas,
            lb[0, :, 0] <= C[0, :, :] @ xfeas,
            C[0, :, :] @ xfeas <= ub[0, :, 0],
            -1 <= xfeas,
            xfeas <= 1,
        ]
        objective = cp.Minimize(jax.random.normal(key[4 + ii], shape=(dim)) @ xfeas)
        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve()

        # Extract RHS parameters
        b = b.at[ii, :, :].set(jnp.array(bfeas.value).reshape((n_eq, 1)))

    # Generate random points-to-be-projected
    xinfeas = jax.random.normal(key[5 + batch_size], (batch_size, dim))

    # Exact projections with QP
    yqp = jnp.zeros((batch_size, dim))
    for ii in range(batch_size):
        yproj = cp.Variable(dim)
        constraints = [
            A[0, :, :] @ yproj == b[ii, :, 0],
            lb[ii, :, 0] <= C[ii, :, :] @ yproj,
            C[ii, :, :] @ yproj <= ub[ii, :, 0],
        ]
        objective = cp.Minimize(cp.sum_squares(yproj - xinfeas[ii, :]))
        problem_qp = cp.Problem(objective=objective, constraints=constraints)
        problem_qp.solve()
        yqp = yqp.at[ii, :].set(jnp.array(yproj.value).reshape((dim)))

    # Projection with JAXopt
    Cqp = jnp.concatenate((C, -C), axis=1)
    d = jnp.concatenate((ub, -lb), axis=1)
    jaxopt_proj = get_jaxopt_projection(
        A=A[0, :, :], C=Cqp[0, :, :], d=d[0, :, 0], dim=dim, tol=1e-3
    )
    xprojjaxopt = jaxopt_proj(xinfeas, b)

    # Projection with cvxpylayer
    cvxpyproj = get_cvxpy_projection(
        A=A[0, :, :], C=Cqp[0, :, :], d=d[0, :, 0], dim=dim
    )
    xprojcvxpyvar = cvxpyproj(
        xinfeas,
        b[:, :, 0],
        solver_args={"verbose": True, "eps_abs": 1e-3, "eps_rel": 1e-3},
    )[0]
    # Projection with PiNet layer
    eq_constraint = EqualityConstraint(A=A, b=b, method="pinv", var_b=True)
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    projection_layer = Project(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint
    )
    xprojiter = projection_layer.call(
        projection_layer.get_init(xinfeas), xinfeas, b, n_iter=100
    )[0]
    # Print the different projections
    idx = 5
    print("Optimal")
    print(yqp[idx])
    print("PiNet")
    print(xprojiter[idx])
    print("JAXopt")
    print(jaxopt_proj(xinfeas, b)[idx])
    print("cvxpylayers")
    print(xprojcvxpyvar[idx])

    # Test the vjps with a simple function
    key_random = jax.random.PRNGKey(seed + 101)
    random_vector = jax.random.normal(key_random, (dim))

    def loss(xinfeas, b, method):
        """Simple loss function to compute gradients."""
        if method == "pinet":
            xproj = projection_layer.call(
                projection_layer.get_init(xinfeas),
                xinfeas,
                b,
                n_iter=500,
                n_iter_bwd=25,
            )[0]
        elif method == "jaxopt":
            xproj = jaxopt_proj(xinfeas, b)
        elif method == "cvxpylayers":
            xproj = cvxpyproj(xinfeas, b[:, :, 0])[0]

        loss = jax.vmap(lambda xx: random_vector @ xx, in_axes=[0])

        return jnp.sum(loss(xproj))

    # Compute and print different gradients
    grad_pinet = jax.grad(loss, argnums=0)(xinfeas, b, "pinet")
    grad_jaxopt = jax.grad(loss, argnums=0)(xinfeas, b, "jaxopt")
    grad_cvxpy = jax.grad(loss, argnums=0)(xinfeas, b, "cvxpylayers")
    print("PiNet")
    print(grad_pinet)
    print("JAXopt")
    print(grad_jaxopt)
    print("cvxpylayers")
    print(grad_cvxpy)
    # Compute the gradient with finite differences
    epsilon = 1e-5
    idx = 2

    def e_i(i, dim):
        """Basis vectors."""
        e = jnp.zeros((1, dim))
        e = e.at[0, i].set(1.0)
        return e

    theloss = loss(xinfeas - epsilon * e_i(idx, dim), b, "pinet")
    theloss_e = loss(xinfeas + epsilon * e_i(idx, dim), b, "pinet")
    grad_fe = (theloss_e - theloss) / (2 * epsilon)
    print(grad_fe)
