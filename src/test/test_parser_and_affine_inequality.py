"""Tests for the parser and affine inequality constraints."""

from itertools import product

import cvxpy as cp
import jax
import jax.numpy as jnp
import pytest

from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.constraints.box import BoxConstraint
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


VALID_METHODS = ["pinv", "cholesky"]
SEEDS = [24, 42]
BATCH_SIZE = [1, 10]


@pytest.mark.parametrize(
    "method, seed, batch_size", product(VALID_METHODS, SEEDS, BATCH_SIZE)
)
def test_simple_2d(method, seed, batch_size):
    # We consider a simple 2D polytope:
    # { x | x_1 = 0, 0<= x_1 + x_2 <= 1 }
    dim = 2
    n_ineq = 1
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


VALID_METHODS = ["pinv", "cholesky"]
SEEDS = [24, 42]
# Note that here batch_size only affects number of projected points
# The same constraints hold throughout the batch
BATCH_SIZE = [1, 10]


@pytest.mark.parametrize(
    "method, seed, batch_size", product(VALID_METHODS, SEEDS, BATCH_SIZE)
)
def test_general_eq_ineq(method, seed, batch_size):
    dim = 100
    n_eq = 50
    n_ineq = 40
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
    (lifted_eq, lifted_box) = parser.parse(method=method)
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
    # Lifted point-to-projected
    xlifted = jnp.concatenate((x.copy(), C @ x), axis=1)
    xk = xlifted.copy()
    for ii in range(n_iter):
        (xk, xlifted) = iteration_step(xk, xlifted)

    yiterated = final_step(xk + xlifted)[:, :dim, :]

    assert jnp.allclose(yqp, yiterated, rtol=1e-3, atol=1e-3)


VALID_METHODS = ["pinv"]
SEEDS = [42]
BATCH_SIZE_VAR = [1, 2]


@pytest.mark.parametrize(
    (
        "method, seed, batch_size_A, batch_size_C, "
        "batch_size_b, batch_size_lb, batch_size_ub, "
        "batch_size_box_lower, batch_size_box_upper, "
        "batch_size_x"
    ),
    product(
        VALID_METHODS,
        SEEDS,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
        BATCH_SIZE_VAR,
    ),
)
def test_general_eq_ineq_box(
    method,
    seed,
    batch_size_A,
    batch_size_C,
    batch_size_b,
    batch_size_lb,
    batch_size_ub,
    batch_size_box_lower,
    batch_size_box_upper,
    batch_size_x,
):
    """This test considers the set:
    A @ x == b,
    l <= C @ x <= u
    lbox <= x[mask] <= ubox
    """
    if batch_size_x < max(
        batch_size_A,
        batch_size_C,
        batch_size_b,
        batch_size_lb,
        batch_size_ub,
        batch_size_box_lower,
        batch_size_box_upper,
    ):
        return

    dim = 20
    n_eq = 10
    n_ineq = 30
    n_box = 5
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=4)
    # Generate equality constraints LHS
    A = jax.random.normal(key[0], shape=(batch_size_A, n_eq, dim))
    # Generate inequality constraints LHS
    C = jax.random.normal(key[1], shape=(batch_size_C, n_ineq, dim))
    # Randomly generate mask for box constraints
    indices = jnp.concatenate([jnp.ones(n_box), jnp.zeros(dim - n_box)])
    mask = jax.random.permutation(key[2], indices).astype(bool)
    # Initialize parameters per batch
    b = jnp.zeros(shape=(batch_size_b, n_eq, 1))
    lb = jnp.zeros(shape=(batch_size_lb, n_ineq, 1))
    ub = jnp.ones(shape=(batch_size_ub, n_ineq, 1))
    box_lower = jnp.zeros(shape=(batch_size_box_lower, n_box, 1))
    box_upper = jnp.zeros(shape=(batch_size_box_upper, n_box, 1))

    # Compute vector parameters of polytope and feasible point
    xfeas = cp.Variable(batch_size_x * dim)
    bfeas = cp.Variable(batch_size_b * n_eq)
    lfeas = cp.Variable(batch_size_lb * n_ineq)
    ufeas = cp.Variable(batch_size_ub * n_ineq)
    lbox = cp.Variable(batch_size_box_lower * n_box)
    ubox = cp.Variable(batch_size_box_upper * n_box)
    constraints = []
    for ii in range(batch_size_x):
        # Define indices for the current batch
        Aidx = min(ii, batch_size_A - 1)
        Cidx = min(ii, batch_size_C - 1)
        bfeasidx = min(ii, batch_size_b - 1)
        lfeasidx = min(ii, batch_size_lb - 1)
        ufeasidx = min(ii, batch_size_ub - 1)
        lboxidx = min(ii, batch_size_box_lower - 1)
        uboxidx = min(ii, batch_size_box_upper - 1)
        # Add constraints
        constraints += [
            A[Aidx, :, :] @ xfeas[ii * dim : (ii + 1) * dim]
            == bfeas[bfeasidx * n_eq : (bfeasidx + 1) * n_eq],
            lfeas[lfeasidx * n_ineq : (lfeasidx + 1) * n_ineq]
            <= C[Cidx, :, :] @ xfeas[ii * dim : (ii + 1) * dim],
            C[Cidx, :, :] @ xfeas[ii * dim : (ii + 1) * dim]
            <= ufeas[ufeasidx * n_ineq : (ufeasidx + 1) * n_ineq],
            -1 <= lbox[lboxidx * n_box : (lboxidx + 1) * n_box],
            ubox[uboxidx * n_box : (uboxidx + 1) * n_box] <= 1,
            lbox[lboxidx * n_box : (lboxidx + 1) * n_box]
            <= xfeas[ii * dim : (ii + 1) * dim][mask],
            xfeas[ii * dim : (ii + 1) * dim][mask]
            <= ubox[uboxidx * n_box : (uboxidx + 1) * n_box],
            xfeas[ii * dim : (ii + 1) * dim] <= 2,
            -2 <= xfeas[ii * dim : (ii + 1) * dim],
        ]
    objective = cp.Minimize(jnp.ones(shape=(dim * batch_size_x)) @ xfeas)
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve(verbose=False)

    # Extract RHS parameters
    b = jnp.array(bfeas.value).reshape((batch_size_b, n_eq, 1))
    lb = jnp.array(lfeas.value).reshape((batch_size_lb, n_ineq, 1))
    ub = jnp.array(ufeas.value).reshape((batch_size_ub, n_ineq, 1))
    box_lower = jnp.array(lbox.value).reshape((batch_size_box_lower, n_box, 1))
    box_upper = jnp.array(ubox.value).reshape((batch_size_box_upper, n_box, 1))

    eq_constraint = EqualityConstraint(A=A, b=b, method=method)
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)
    box_constraint = BoxConstraint(
        lower_bound=box_lower, upper_bound=box_upper, mask=mask
    )

    # Parse constraints
    parser = ConstraintParser(
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        box_constraint=box_constraint,
    )
    (lifted_eq, lifted_box) = parser.parse(method=method)

    # Point to be projected
    x = jax.random.uniform(key[3], shape=(batch_size_x, dim, 1), minval=-3, maxval=3)

    # Compute the projection by solving QP
    yqp = jnp.zeros(shape=(batch_size_x, dim, 1))
    for ii in range(batch_size_x):
        # Define indices for batch
        Aidx = min(ii, batch_size_A - 1)
        Cidx = min(ii, batch_size_C - 1)
        bfeasidx = min(ii, batch_size_b - 1)
        lfeasidx = min(ii, batch_size_lb - 1)
        ufeasidx = min(ii, batch_size_ub - 1)
        lboxidx = min(ii, batch_size_box_lower - 1)
        uboxidx = min(ii, batch_size_box_upper - 1)
        yproj = cp.Variable(dim)
        constraints = [
            A[Aidx, :, :] @ yproj == b[bfeasidx, :, 0],
            lb[lfeasidx, :, 0] <= C[Cidx, :, :] @ yproj,
            C[Cidx, :, :] @ yproj <= ub[ufeasidx, :, 0],
            box_lower[lboxidx, :, 0] <= yproj[mask],
            yproj[mask] <= box_upper[uboxidx, :, 0],
        ]
        objective = cp.Minimize(cp.sum_squares(yproj - x[ii, :, 0]))
        problem_qp = cp.Problem(objective=objective, constraints=constraints)
        problem_qp.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7, verbose=False)
        yqp = yqp.at[ii, :, :].set(jnp.array(yproj.value).reshape((dim, 1)))

    # Compute the projection with QP, but in lifted form
    ylifted = jnp.zeros(shape=(batch_size_x, dim, 1))
    for ii in range(batch_size_x):
        # Define indices for batch
        # Should be careful here, because of the lifting
        ACidx = min(ii, max(batch_size_A - 1, batch_size_C - 1))
        bfeasidx = min(ii, batch_size_b - 1)
        loweridx = min(ii, max(batch_size_lb - 1, batch_size_box_lower - 1))
        upperidx = min(ii, max(batch_size_ub - 1, batch_size_box_upper - 1))
        yliftedproj = cp.Variable(dim + n_ineq)
        constraints_lifted = [
            lifted_eq.A[ACidx, :, :] @ yliftedproj == lifted_eq.b[bfeasidx, :, 0],
            lifted_box.lower_bound[loweridx, :, 0] <= yliftedproj[lifted_box.mask],
            yliftedproj[lifted_box.mask] <= lifted_box.upper_bound[upperidx, :, 0],
        ]
        objective_lifted = cp.Minimize(cp.sum_squares(yliftedproj[:dim] - x[ii, :, 0]))
        problem_lifted = cp.Problem(
            objective=objective_lifted, constraints=constraints_lifted
        )
        problem_lifted.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7, verbose=False)
        ylifted = ylifted.at[ii, :, :].set(
            jnp.array(yliftedproj.value[:dim]).reshape((dim, 1))
        )

    assert jnp.allclose(yqp, ylifted, rtol=1e-5, atol=1e-5)

    # Compute with iterative using lifting of:
    # Equality + Inequality + Box
    n_iter = 5000
    (iteration_step, final_step) = build_iteration_step(
        lifted_eq, lifted_box, n_ineq, dim
    )
    xlifted = jnp.concatenate((x.copy(), C @ x), axis=1)
    xk = xlifted.copy()
    for ii in range(n_iter):
        (xk, xlifted) = iteration_step(xk, xlifted)

    yiterated = final_step(xk + xlifted)[:, :dim, :]

    assert jnp.allclose(yqp, yiterated, rtol=1e-3, atol=1e-3)
    # Compute with iterative using lifting of:
    # Equality + Inequality
    # Write box constraints as affine inequality constraints
    Caug = jnp.concatenate(
        (C, jnp.tile(jnp.eye(dim)[mask, :].reshape(1, n_box, dim), (C.shape[0], 1, 1))),
        axis=1,
    )
    # Adapt lower and upper bounds accordingly
    # Maximum batch size of lower and upper bound
    mblb = max(lb.shape[0], box_lower.shape[0])
    mbub = max(ub.shape[0], box_upper.shape[0])
    lbaug = jnp.concatenate(
        (
            jnp.tile(lb, (mblb // lb.shape[0], 1, 1)),
            jnp.tile(box_lower, (mblb // box_lower.shape[0], 1, 1)),
        ),
        axis=1,
    )
    ubaug = jnp.concatenate(
        (
            jnp.tile(ub, (mbub // ub.shape[0], 1, 1)),
            jnp.tile(box_upper, (mbub // box_upper.shape[0], 1, 1)),
        ),
        axis=1,
    )
    n_ineq_aug = n_ineq + n_box
    ineq_constraint_aug = AffineInequalityConstraint(C=Caug, lb=lbaug, ub=ubaug)

    parser_aug = ConstraintParser(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint_aug
    )

    (lifted_eq_aug, lifted_box_aug) = parser_aug.parse()

    n_iter = 5000
    (iteration_step_aug, final_step_aug) = build_iteration_step(
        lifted_eq_aug, lifted_box_aug, n_ineq_aug, dim
    )
    xlifted = jnp.concatenate((x.copy(), Caug @ x), axis=1)
    xk = xlifted.copy()
    for ii in range(n_iter):
        (xk, xlifted) = iteration_step_aug(xk, xlifted)

    yiterated_aug = final_step_aug(xk + xlifted)[:, :dim, :]

    assert jnp.allclose(yqp, yiterated_aug, rtol=1e-3, atol=1e-3)
