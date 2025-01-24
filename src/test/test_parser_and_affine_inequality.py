import cvxpy as cp
import jax
import jax.numpy as jnp

from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.constraints.constraint_parser import ConstraintParser

jax.config.update("jax_enable_x64", True)


# One iteration of the iterated projection
# TODO: Replace this with a structured implementation,
# that probably consumes the output of the parser.
def build_iteration_step(lifted_eq, lifted_box, n_ineq, dim):
    def iteration_step(xk, xlifted, alpha=0.9, beta=0.8):
        qktilde = lifted_box.project(xk + xlifted)
        xlifted = xlifted.at[:, -n_ineq:, :].set(qktilde[:, dim:, :])
        qk = 2 * beta * (qktilde - xlifted) - xk
        pk = 2 * beta * (lifted_eq.project(qk + xlifted) - xlifted) - qk
        xk = (1 - alpha) * xk + alpha * pk
        return (xk, xlifted)

    # The second element is used to extract the projection from the auxiliary
    return (jax.jit(iteration_step), jax.jit(lifted_box.project))


def test_simple_2d():
    # We consider a simple 2D polytope:
    # { x | x_1 = 0, 0<= x_1 + x_2 <= 1 }
    dim = 2
    n_ineq = 1
    method = "pinv"
    batch_size = 10
    seed = 42
    key = jax.random.PRNGKey(seed)
    # Equality constraint: A @ x = b
    A = jnp.array([[[1, 0]]])
    b = jnp.zeros(shape=(1, 1, 1))
    eq_constraint = EqualityConstraint(A=A, b=b, method=method)
    # Inequality constraint: l <= C @ x <= u
    C = jnp.array([[[1, 1]]])
    lb = jnp.zeros(shape=(1, 1, 1))
    ub = jnp.ones(shape=(1, 1, 1))
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    # Parse constraints
    parser = ConstraintParser(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint
    )
    (lifted_eq, lifted_box) = parser.parse()

    # Point to be projected
    x = jax.random.uniform(key, shape=(batch_size, dim, 1), minval=-2, maxval=2)

    # Compute the projection in closed form
    yclosed = jnp.concatenate(
        (
            jnp.zeros(shape=(batch_size, 1, 1)),
            jnp.clip(x[:, 1, :], lb, ub).reshape(batch_size, 1, 1),
        ),
        axis=1,
    )

    # Compute the projection with QP
    for ii in range(batch_size):
        ycp = cp.Variable(dim)
        constraints = [
            A[0, :, :] @ ycp == b[0, :, 0],
            lb[0, :, 0] <= C[0, :, :] @ ycp,
            C[0, :, :] @ ycp <= ub[0, :, 0],
        ]
        objective = cp.Minimize(cp.sum_squares(ycp - x[ii, :, 0]))
        problem_exact = cp.Problem(objective=objective, constraints=constraints)
        problem_exact.solve()
        # Extract true projection
        y = jnp.reshape(jnp.array(ycp.value), shape=(1, 2, 1))
        assert jnp.allclose(y, yclosed[ii, :])

    # Compute the projection with QP, but in lifted form
    # Last n_ineq variables corresponding to inequality lifting
    for ii in range(batch_size):
        yliftedcp = cp.Variable(dim + n_ineq)
        constraints_lifted = [
            lifted_eq.A[0, :, :] @ yliftedcp == lifted_eq.b[0, :, 0],
            lifted_box.lower_bound[0, :, 0] <= yliftedcp[lifted_box.mask],
            yliftedcp[lifted_box.mask] <= lifted_box.upper_bound[0, :, 0],
        ]
        objective_lifted = cp.Minimize(cp.sum_squares(yliftedcp[:dim] - x[ii, :, 0]))
        problem_lifted = cp.Problem(
            objective=objective_lifted, constraints=constraints_lifted
        )
        problem_lifted.solve()
        # Extract lifted projection
        ylifted = jnp.expand_dims(jnp.array(yliftedcp.value[:dim]), axis=1)
        assert jnp.allclose(ylifted, yclosed[ii, :])

    # Compute the projection with iterative
    n_iter = 100
    (iteration_step, final_step) = build_iteration_step(
        lifted_eq, lifted_box, n_ineq, dim
    )
    xlifted = jnp.concatenate(
        (x.copy(), jnp.zeros(shape=(batch_size, n_ineq, 1))), axis=1
    )
    xk = xlifted.copy()
    for ii in range(n_iter):
        (xk, xlifted) = iteration_step(xk, xlifted)

    yiterated = final_step(xk + xlifted)[:, :dim, :]

    assert jnp.allclose(yclosed, yiterated, rtol=1e-6, atol=1e-6)


def test_general_eq_ineq():
    dim = 100
    n_eq = 50
    n_ineq = 50
    method = "pinv"
    seed = 42
    batch_size = 50
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=3)
    # Generate equality constraints LHS
    A = jax.random.normal(key[0], shape=(1, n_eq, dim))
    # Generate inequality constraints LHS
    C = jax.random.normal(key[1], shape=(1, n_ineq, dim))
    # Compute RHS by solving feasibility problem
    xfeas = cp.Variable(dim)
    bfeas = cp.Variable(n_eq)
    lfeas = cp.Variable(n_ineq)
    ufeas = cp.Variable(n_ineq)
    constraints = [
        A[0, :, :] @ xfeas == bfeas,
        lfeas <= C[0, :, :] @ xfeas,
        C[0, :, :] @ xfeas <= ufeas,
        -1 <= xfeas,
        xfeas <= 1,
    ]
    objective = cp.Minimize(jnp.ones(shape=(dim)) @ xfeas)
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve()
    # Extract RHS parameters
    b = jnp.tile(jnp.array(bfeas.value).reshape((1, n_eq, 1)), (1, 1, 1))
    lb = jnp.tile(jnp.array(lfeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))
    ub = jnp.tile(jnp.array(ufeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))

    eq_constraint = EqualityConstraint(A=A, b=b, method=method)
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    # Parse constraints
    parser = ConstraintParser(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint
    )
    (lifted_eq, lifted_box) = parser.parse()
    # TODO: Make this batched
    # Point to be projected
    x = jax.random.uniform(key[2], shape=(batch_size, dim, 1), minval=-2, maxval=2)

    # Compute the projection by solving QP
    yqp = jnp.zeros(shape=(batch_size, dim, 1))
    for ii in range(batch_size):
        yproj = cp.Variable(dim)
        constraints = [
            A[0, :, :] @ yproj == b[0, :, 0],
            lb[0, :, 0] <= C[0, :, :] @ yproj,
            C[0, :, :] @ yproj <= ub[0, :, 0],
        ]
        objective = cp.Minimize(cp.sum_squares(yproj - x[ii, :, 0]))
        problem_qp = cp.Problem(objective=objective, constraints=constraints)
        problem_qp.solve()
        yqp = yqp.at[ii, :, :].set(jnp.array(yproj.value).reshape((dim, 1)))

    # Compute the projection with QP, but in lifted form
    ylifted = jnp.zeros(shape=(batch_size, dim, 1))
    for ii in range(batch_size):
        yliftedproj = cp.Variable(dim + n_ineq)
        constraints_lifted = [
            lifted_eq.A[0, :, :] @ yliftedproj == lifted_eq.b[0, :, 0],
            lifted_box.lower_bound[0, :, 0] <= yliftedproj[lifted_box.mask],
            yliftedproj[lifted_box.mask] <= lifted_box.upper_bound[0, :, 0],
        ]
        objective_lifted = cp.Minimize(cp.sum_squares(yliftedproj[:dim] - x[ii, :, 0]))
        problem_lifted = cp.Problem(
            objective=objective_lifted, constraints=constraints_lifted
        )
        problem_lifted.solve()
        ylifted = ylifted.at[ii, :, :].set(
            jnp.array(yliftedproj.value[:dim]).reshape((dim, 1))
        )

    assert jnp.allclose(yqp, ylifted, rtol=1e-6, atol=1e-6)

    # Compute the projection with iterative
    n_iter = 500
    (iteration_step, final_step) = build_iteration_step(
        lifted_eq, lifted_box, n_ineq, dim
    )
    xlifted = jnp.concatenate((x.copy(), C @ x), axis=1)
    xk = xlifted.copy()
    for ii in range(n_iter):
        (xk, xlifted) = iteration_step(xk, xlifted)

    yiterated = final_step(xk + xlifted)[:, :dim, :]

    assert jnp.allclose(yqp, yiterated, rtol=1e-3, atol=1e-3)


# Here the parser should handle all of:
# Equality, Inequality and Box constraint
def test_general_eq_ineq_box():
    return
