"""Tests for the parser and affine inequality constraints."""

import re
from itertools import product
from typing import cast

import cvxpy as cp
import jax
import jax.numpy as jnp
import pytest

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    BoxConstraintSpecification,
    ConstraintParser,
    EqualityConstraint,
    ProjectionInstance,
    build_iteration_step,
)

jax.config.update("jax_enable_x64", True)

VALID_METHODS = ["pinv"]
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
    # Equality constraint: a_mat @ x = b
    a_mat = jnp.array([[[1, 0]]])
    b = jnp.zeros(shape=(1, 1, 1))
    eq_constraint = EqualityConstraint(a_mat=a_mat, b=b, method=method)
    # Inequality constraint: l <= c_mat @ x <= u
    c_mat = jnp.array([[[1, 1]]])
    lb = jnp.zeros(shape=(1, 1, 1))
    ub = jnp.ones(shape=(1, 1, 1))
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)

    # Parse constraints
    parser = ConstraintParser(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint
    )
    (lifted_eq, lifted_box, _) = parser.parse()
    assert lifted_eq is not None, (
        "Parser should return a lifted equality constraint for mixed equality and "
        "inequality constraints."
    )
    assert isinstance(lifted_box, BoxConstraint), (
        "Parser should return a lifted box constraint for mixed equality and "
        "inequality constraints."
    )
    assert lifted_box.lb is not None, (
        "Lifted box constraint should expose lower bounds after parsing."
    )
    assert lifted_box.ub is not None, (
        "Lifted box constraint should expose upper bounds after parsing."
    )

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
        constraints = cast(
            list[cp.Constraint],
            [
                a_mat[0, :, :] @ ycp == b[0, :, 0],
                lb[0, :, 0] <= c_mat[0, :, :] @ ycp,
                c_mat[0, :, :] @ ycp <= ub[0, :, 0],
            ],
        )
        objective = cp.Minimize(cp.sum_squares(ycp - x[ii, :, 0]))
        problem_exact = cp.Problem(objective=objective, constraints=constraints)
        problem_exact.solve()
        # Extract true projection
        y = jnp.reshape(jnp.array(ycp.value), shape=(1, 2, 1))
        assert jnp.allclose(y, yclosed[ii, :]), (
            "Closed-form projection should match the direct QP solution. "
            f"Batch {ii}: expected {yclosed[ii, :]}, got {y}."
        )

    # Compute the projection with QP, but in lifted form
    # Last n_ineq variables corresponding to inequality lifting
    for ii in range(batch_size):
        yliftedcp = cp.Variable(dim + n_ineq)
        constraints_lifted = cast(
            list[cp.Constraint],
            [
                lifted_eq.a_mat[0, :, :] @ yliftedcp == lifted_eq.b[0, :, 0],
                lifted_box.lb[0, :, 0] <= yliftedcp[lifted_box.mask],
                yliftedcp[lifted_box.mask] <= lifted_box.ub[0, :, 0],
            ],
        )
        objective_lifted = cp.Minimize(cp.sum_squares(yliftedcp[:dim] - x[ii, :, 0]))
        problem_lifted = cp.Problem(
            objective=objective_lifted, constraints=constraints_lifted
        )
        problem_lifted.solve()
        # Extract lifted projection
        assert yliftedcp.value is not None, (
            "Lifted QP solve should return a primal solution."
        )
        ylifted = jnp.expand_dims(jnp.array(yliftedcp.value[:dim]), axis=1)
        assert jnp.allclose(ylifted, yclosed[ii, :]), (
            "Lifted QP projection should match the closed-form solution. "
            f"Batch {ii}: expected {yclosed[ii, :]}, got {ylifted}."
        )

    # Compute the projection with iterative
    n_iter = 200
    (iteration_step, final_step) = build_iteration_step(lifted_eq, lifted_box, dim)
    sk = ProjectionInstance(x=jnp.zeros(shape=(batch_size, dim + n_ineq, 1)))
    for _ii in range(n_iter):
        sk = iteration_step(sk, ProjectionInstance(x=x), 0.1, 1.0)

    yiterated = final_step(sk).x[:, :dim, :]

    assert jnp.allclose(yclosed, yiterated, rtol=1e-6, atol=1e-6), (
        "Iterative lifted projection should match the closed-form solution. "
        f"Expected {yclosed}, got {yiterated}."
    )


# Note that here batch_size only affects number of projected points
# The same constraints hold throughout the batch


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
    a_mat = jax.random.normal(key[0], shape=(1, n_eq, dim))
    # Generate inequality constraints LHS
    c_mat = jax.random.normal(key[1], shape=(1, n_ineq, dim))
    # Compute RHS by solving feasibility problem
    xfeas = cp.Variable(dim)
    bfeas = cp.Variable(n_eq)
    lfeas = cp.Variable(n_ineq)
    ufeas = cp.Variable(n_ineq)
    constraints = cast(
        list[cp.Constraint],
        [
            a_mat[0, :, :] @ xfeas == bfeas,
            lfeas <= c_mat[0, :, :] @ xfeas,
            c_mat[0, :, :] @ xfeas <= ufeas,
            -1 <= xfeas,
            xfeas <= 1,
        ],
    )
    objective = cp.Minimize(jnp.ones(shape=(dim)) @ xfeas)
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve()
    # Extract RHS parameters
    b = jnp.tile(jnp.array(bfeas.value).reshape((1, n_eq, 1)), (1, 1, 1))
    lb = jnp.tile(jnp.array(lfeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))
    ub = jnp.tile(jnp.array(ufeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))

    eq_constraint = EqualityConstraint(a_mat=a_mat, b=b, method=method)
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)

    # Parse constraints
    parser = ConstraintParser(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint
    )
    (lifted_eq, lifted_box, _) = parser.parse(method=method)
    assert lifted_eq is not None, (
        "Parser should return a lifted equality constraint for the general "
        "equality-plus-inequality case."
    )
    assert isinstance(lifted_box, BoxConstraint), (
        "Parser should return a lifted box constraint for the general "
        "equality-plus-inequality case."
    )
    assert lifted_box.lb is not None, (
        "Lifted box constraint should include lower bounds in the general case."
    )
    assert lifted_box.ub is not None, (
        "Lifted box constraint should include upper bounds in the general case."
    )
    # Point to be projected
    x = jax.random.uniform(key[2], shape=(batch_size, dim, 1), minval=-2, maxval=2)

    # Compute the projection by solving QP
    yqp = jnp.zeros(shape=(batch_size, dim, 1))
    for ii in range(batch_size):
        yproj = cp.Variable(dim)
        constraints = cast(
            list[cp.Constraint],
            [
                a_mat[0, :, :] @ yproj == b[0, :, 0],
                lb[0, :, 0] <= c_mat[0, :, :] @ yproj,
                c_mat[0, :, :] @ yproj <= ub[0, :, 0],
            ],
        )
        objective = cp.Minimize(cp.sum_squares(yproj - x[ii, :, 0]))
        problem_qp = cp.Problem(objective=objective, constraints=constraints)
        problem_qp.solve()
        yqp = yqp.at[ii, :, :].set(jnp.array(yproj.value).reshape((dim, 1)))

    # Compute the projection with QP, but in lifted form
    ylifted = jnp.zeros(shape=(batch_size, dim, 1))
    for ii in range(batch_size):
        yliftedproj = cp.Variable(dim + n_ineq)
        constraints_lifted = cast(
            list[cp.Constraint],
            [
                lifted_eq.a_mat[0, :, :] @ yliftedproj == lifted_eq.b[0, :, 0],
                lifted_box.lb[0, :, 0] <= yliftedproj[lifted_box.mask],
                yliftedproj[lifted_box.mask] <= lifted_box.ub[0, :, 0],
            ],
        )
        objective_lifted = cp.Minimize(cp.sum_squares(yliftedproj[:dim] - x[ii, :, 0]))
        problem_lifted = cp.Problem(
            objective=objective_lifted, constraints=constraints_lifted
        )
        problem_lifted.solve()
        assert yliftedproj.value is not None, (
            "Lifted QP solve should return a primal solution in the general case."
        )
        ylifted = ylifted.at[ii, :, :].set(
            jnp.array(yliftedproj.value[:dim]).reshape((dim, 1))
        )

    assert jnp.allclose(yqp, ylifted, rtol=1e-6, atol=1e-6), (
        "Lifted QP projection should match the original QP solution in the "
        f"general case. Expected {yqp}, got {ylifted}."
    )

    # Compute the projection with iterative
    n_iter = 500
    (iteration_step, final_step) = build_iteration_step(lifted_eq, lifted_box, dim)
    iteration_step = jax.jit(iteration_step)
    xk = ProjectionInstance(x=jnp.zeros(shape=(batch_size, dim + n_ineq, 1)))
    for _ii in range(n_iter):
        xk = iteration_step(xk, ProjectionInstance(x=x), sigma=1.0, omega=1.0)

    yiterated = final_step(xk).x[:, :dim, :]

    assert jnp.allclose(yqp, yiterated, rtol=1e-3, atol=1e-3), (
        "Iterative lifted projection should match the original QP solution in "
        f"the general case. Expected {yqp}, got {yiterated}."
    )


SEEDS_VAR = [42]
BATCH_SIZE_VAR = [1, 2]


@pytest.mark.parametrize(
    (
        "method, seed, batch_size_a, batch_size_c, "
        "batch_size_b, batch_size_lb, batch_size_ub, "
        "batch_size_box_lower, batch_size_box_upper, "
        "batch_size_x"
    ),
    product(
        VALID_METHODS,
        SEEDS_VAR,
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
    method: str,
    seed: int,
    batch_size_a: int,
    batch_size_c: int,
    batch_size_b: int,
    batch_size_lb: int,
    batch_size_ub: int,
    batch_size_box_lower: int,
    batch_size_box_upper: int,
    batch_size_x: int,
) -> None:
    """This test considers the set:
    a_mat @ x == b,
    l <= c_mat @ x <= u
    lbox <= x[mask] <= ubox

    Args:
        method: Projection method to test.
        seed: Random seed.
        batch_size_a: Batch size for equality constraint matrices.
        batch_size_c: Batch size for inequality constraint matrices.
        batch_size_b: Batch size for equality constraint vectors.
        batch_size_lb: Batch size for inequality lower bounds.
        batch_size_ub: Batch size for inequality upper bounds.
        batch_size_box_lower: Batch size for box lower bounds.
        batch_size_box_upper: Batch size for box upper bounds.
        batch_size_x: Batch size for points to project.

    Returns:
        None.
    """
    if batch_size_x < max(
        batch_size_a,
        batch_size_c,
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
    lower_feas_bound = -2
    upper_feas_bound = 2
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=4)
    # Generate equality constraints LHS
    a_mat = jax.random.normal(key[0], shape=(batch_size_a, n_eq, dim))
    # Generate inequality constraints LHS
    c_mat = jax.random.normal(key[1], shape=(batch_size_c, n_ineq, dim))
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
        aidx = min(ii, batch_size_a - 1)
        cidx = min(ii, batch_size_c - 1)
        bfeasidx = min(ii, batch_size_b - 1)
        lfeasidx = min(ii, batch_size_lb - 1)
        ufeasidx = min(ii, batch_size_ub - 1)
        lboxidx = min(ii, batch_size_box_lower - 1)
        uboxidx = min(ii, batch_size_box_upper - 1)
        # Add constraints
        constraints += [
            a_mat[aidx, :, :] @ xfeas[ii * dim : (ii + 1) * dim]
            == bfeas[bfeasidx * n_eq : (bfeasidx + 1) * n_eq],
            lfeas[lfeasidx * n_ineq : (lfeasidx + 1) * n_ineq]
            <= c_mat[cidx, :, :] @ xfeas[ii * dim : (ii + 1) * dim],
            c_mat[cidx, :, :] @ xfeas[ii * dim : (ii + 1) * dim]
            <= ufeas[ufeasidx * n_ineq : (ufeasidx + 1) * n_ineq],
            -1 <= lbox[lboxidx * n_box : (lboxidx + 1) * n_box],
            ubox[uboxidx * n_box : (uboxidx + 1) * n_box] <= 1,
            lbox[lboxidx * n_box : (lboxidx + 1) * n_box]
            <= xfeas[ii * dim : (ii + 1) * dim][mask],
            xfeas[ii * dim : (ii + 1) * dim][mask]
            <= ubox[uboxidx * n_box : (uboxidx + 1) * n_box],
            xfeas[ii * dim : (ii + 1) * dim] <= upper_feas_bound,
            lower_feas_bound <= xfeas[ii * dim : (ii + 1) * dim],
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

    eq_constraint = EqualityConstraint(a_mat=a_mat, b=b, method=method)
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)
    box_constraint = BoxConstraint(
        BoxConstraintSpecification(lb=box_lower, ub=box_upper, mask=mask)
    )

    # Parse constraints
    parser = ConstraintParser(
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        box_constraint=box_constraint,
    )
    (lifted_eq, lifted_box, _) = parser.parse(method=method)
    assert lifted_eq is not None, (
        "Parser should return a lifted equality constraint for mixed equality, "
        "inequality, and box constraints."
    )
    assert isinstance(lifted_box, BoxConstraint), (
        "Parser should return a lifted box constraint for mixed equality, "
        "inequality, and box constraints."
    )
    assert lifted_box.lb is not None, (
        "Lifted box constraint should include lower bounds for the mixed case."
    )
    assert lifted_box.ub is not None, (
        "Lifted box constraint should include upper bounds for the mixed case."
    )

    # Point to be projected
    x = jax.random.uniform(key[3], shape=(batch_size_x, dim, 1), minval=-3, maxval=3)

    # Compute the projection by solving QP
    yqp = jnp.zeros(shape=(batch_size_x, dim, 1))
    for ii in range(batch_size_x):
        # Define indices for batch
        aidx = min(ii, batch_size_a - 1)
        cidx = min(ii, batch_size_c - 1)
        bfeasidx = min(ii, batch_size_b - 1)
        lfeasidx = min(ii, batch_size_lb - 1)
        ufeasidx = min(ii, batch_size_ub - 1)
        lboxidx = min(ii, batch_size_box_lower - 1)
        uboxidx = min(ii, batch_size_box_upper - 1)
        yproj = cp.Variable(dim)
        constraints = cast(
            list[cp.Constraint],
            [
                a_mat[aidx, :, :] @ yproj == b[bfeasidx, :, 0],
                lb[lfeasidx, :, 0] <= c_mat[cidx, :, :] @ yproj,
                c_mat[cidx, :, :] @ yproj <= ub[ufeasidx, :, 0],
                box_lower[lboxidx, :, 0] <= yproj[mask],
                yproj[mask] <= box_upper[uboxidx, :, 0],
            ],
        )
        objective = cp.Minimize(cp.sum_squares(yproj - x[ii, :, 0]))
        problem_qp = cp.Problem(objective=objective, constraints=constraints)
        problem_qp.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7, verbose=False)
        yqp = yqp.at[ii, :, :].set(jnp.array(yproj.value).reshape((dim, 1)))

    # Compute the projection with QP, but in lifted form
    ylifted = jnp.zeros(shape=(batch_size_x, dim, 1))
    for ii in range(batch_size_x):
        # Define indices for batch
        # Should be careful here, because of the lifting
        acidx = min(ii, max(batch_size_a - 1, batch_size_c - 1))
        bfeasidx = min(ii, batch_size_b - 1)
        loweridx = min(ii, max(batch_size_lb - 1, batch_size_box_lower - 1))
        upperidx = min(ii, max(batch_size_ub - 1, batch_size_box_upper - 1))
        yliftedproj = cp.Variable(dim + n_ineq)
        constraints_lifted = cast(
            list[cp.Constraint],
            [
                lifted_eq.a_mat[acidx, :, :] @ yliftedproj == lifted_eq.b[bfeasidx, :, 0],
                lifted_box.lb[loweridx, :, 0] <= yliftedproj[lifted_box.mask],
                yliftedproj[lifted_box.mask] <= lifted_box.ub[upperidx, :, 0],
            ],
        )
        objective_lifted = cp.Minimize(cp.sum_squares(yliftedproj[:dim] - x[ii, :, 0]))
        problem_lifted = cp.Problem(
            objective=objective_lifted, constraints=constraints_lifted
        )
        problem_lifted.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7, verbose=False)
        assert yliftedproj.value is not None, (
            "Lifted QP solve should return a primal solution for the mixed case."
        )
        ylifted = ylifted.at[ii, :, :].set(
            jnp.array(yliftedproj.value[:dim]).reshape((dim, 1))
        )

    assert jnp.allclose(yqp, ylifted, rtol=1e-5, atol=1e-5), (
        "Lifted QP projection should match the original QP solution for mixed "
        f"equality, inequality, and box constraints. Expected {yqp}, got {ylifted}."
    )

    # Compute with iterative using lifting of:
    # Equality + Inequality + Box
    n_iter = 5000
    (iteration_step, final_step) = build_iteration_step(lifted_eq, lifted_box, dim)
    iteration_step = jax.jit(iteration_step)
    xk = ProjectionInstance(x=jnp.zeros(shape=(batch_size_x, dim + n_ineq, 1)))
    for _ii in range(n_iter):
        xk = iteration_step(xk, ProjectionInstance(x=x), sigma=1.0, omega=1.0)

    yiterated = final_step(xk).x[:, :dim, :]

    assert jnp.allclose(yqp, yiterated, rtol=1e-3, atol=1e-3), (
        "Iterative lifted projection should match the original QP solution for "
        "mixed equality, inequality, and box constraints. "
        f"Expected {yqp}, got {yiterated}."
    )
    # Compute with iterative using lifting of:
    # Equality + Inequality
    # Write box constraints as affine inequality constraints
    c_aug = jnp.concatenate(
        (
            c_mat,
            jnp.tile(
                jnp.eye(dim)[mask, :].reshape(1, n_box, dim),
                (c_mat.shape[0], 1, 1),
            ),
        ),
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
    ineq_constraint_aug = AffineInequalityConstraint(c_mat=c_aug, lb=lbaug, ub=ubaug)

    parser_aug = ConstraintParser(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint_aug
    )

    (lifted_eq, lifted_box, _) = parser_aug.parse()
    assert lifted_eq is not None, (
        "Parser should return a lifted equality constraint after augmenting box "
        "constraints as affine inequalities."
    )
    assert isinstance(lifted_box, BoxConstraint), (
        "Parser should return a lifted box constraint after augmenting box "
        "constraints as affine inequalities."
    )
    assert lifted_box.lb is not None, (
        "Augmented lifted box constraint should include lower bounds."
    )
    assert lifted_box.ub is not None, (
        "Augmented lifted box constraint should include upper bounds."
    )

    n_iter = 5000
    (iteration_step, final_step) = build_iteration_step(lifted_eq, lifted_box, dim)
    iteration_step = jax.jit(iteration_step)
    xk = ProjectionInstance(x=jnp.zeros(shape=(batch_size_x, dim + n_ineq_aug, 1)))
    for _ii in range(n_iter):
        xk = iteration_step(xk, ProjectionInstance(x=x), sigma=1.0, omega=1.0)

    yiterated = final_step(xk).x[:, :dim, :]

    assert jnp.allclose(yqp, yiterated, rtol=1e-3, atol=1e-3), (
        "Iterative projection with augmented affine inequalities should match "
        f"the original QP solution. Expected {yqp}, got {yiterated}."
    )


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZE))
def test_simple_no_equality(seed, batch_size):
    # We consider a simple 2D polytope:
    # { x | 0 <= x_1 + x_2 <= 1 }
    dim = 2
    n_ineq = 1
    key = jax.random.PRNGKey(seed)
    # Inequality constraint: l <= c_mat @ x <= u
    c_mat = jnp.array([[[1, 1]]])
    lb = jnp.zeros(shape=(1, 1, 1))
    ub = jnp.ones(shape=(1, 1, 1))
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)

    # Parse constraints
    parser = ConstraintParser(eq_constraint=None, ineq_constraint=ineq_constraint)
    (lifted_eq, lifted_box, _) = parser.parse()
    assert lifted_eq is not None, (
        "Parser should synthesize a lifted equality constraint even when the "
        "original problem has no equality constraints."
    )
    assert isinstance(lifted_box, BoxConstraint), (
        "Parser should synthesize a lifted box constraint for pure inequality "
        "constraints."
    )
    assert lifted_box.lb is not None, (
        "Lifted box constraint should expose lower bounds for the pure inequality case."
    )
    assert lifted_box.ub is not None, (
        "Lifted box constraint should expose upper bounds for the pure inequality case."
    )

    # Point to be projected
    x = jax.random.uniform(key, shape=(batch_size, dim, 1), minval=-2, maxval=2)

    # Compute the projection with iterative
    (lifted_eq, lifted_box, _) = parser.parse()
    assert lifted_eq is not None, (
        "Parser should keep returning a lifted equality constraint on repeated "
        "parses of the pure inequality case."
    )
    assert isinstance(lifted_box, BoxConstraint), (
        "Parser should keep returning a lifted box constraint on repeated parses "
        "of the pure inequality case."
    )
    assert lifted_box.lb is not None, (
        "Repeated parse should preserve lifted lower bounds in the pure inequality case."
    )
    assert lifted_box.ub is not None, (
        "Repeated parse should preserve lifted upper bounds in the pure inequality case."
    )

    n_iter = 500
    (iteration_step, final_step) = build_iteration_step(lifted_eq, lifted_box, dim)
    xk = ProjectionInstance(x=jnp.zeros(shape=(batch_size, dim + n_ineq, 1)))
    for _ii in range(n_iter):
        xk = iteration_step(xk, ProjectionInstance(x=x), 0.1, 1.0)

    yiterated = final_step(xk).x[:, :dim, :]

    # Compute the projection with QP
    for ii in range(batch_size):
        ycp = cp.Variable(dim)
        constraints = cast(
            list[cp.Constraint],
            [
                lb[0, :, 0] <= c_mat[0, :, :] @ ycp,
                c_mat[0, :, :] @ ycp <= ub[0, :, 0],
            ],
        )
        objective = cp.Minimize(cp.sum_squares(ycp - x[ii, :, 0]))
        problem_exact = cp.Problem(objective=objective, constraints=constraints)
        problem_exact.solve()
        # Extract true projection
        y_qp = jnp.reshape(jnp.array(ycp.value), shape=(1, 2, 1))

        # Compute the projection with QP, but in lifted form
        # Last n_ineq variables corresponding to inequality lifting
        yliftedcp = cp.Variable(dim + n_ineq)
        constraints_lifted = cast(
            list[cp.Constraint],
            [
                lifted_eq.a_mat[0, :, :] @ yliftedcp == lifted_eq.b[0, :, 0],
                lifted_box.lb[0, :, 0] <= yliftedcp[lifted_box.mask],
                yliftedcp[lifted_box.mask] <= lifted_box.ub[0, :, 0],
            ],
        )
        objective_lifted = cp.Minimize(cp.sum_squares(yliftedcp[:dim] - x[ii, :, 0]))
        problem_lifted = cp.Problem(
            objective=objective_lifted, constraints=constraints_lifted
        )
        problem_lifted.solve()
        # Extract lifted projection
        assert yliftedcp.value is not None, (
            "Lifted QP solve should return a primal solution in the pure inequality case."
        )
        ylifted = jnp.expand_dims(jnp.array(yliftedcp.value[:dim]), axis=1)

        # Check the projections match
        assert jnp.allclose(ylifted, y_qp[0, :, :]), (
            "Lifted QP projection should match the original QP solution in the "
            f"pure inequality case. Batch {ii}: expected {y_qp[0, :, :]}, got {ylifted}."
        )
        assert jnp.allclose(y_qp[0, :, :], yiterated[ii, :, :], rtol=1e-6, atol=1e-6), (
            "Iterative lifted projection should match the original QP solution in "
            "the pure inequality case. "
            f"Batch {ii}: expected {y_qp[0, :, :]}, got {yiterated[ii, :, :]}."
        )


def test_affine_inequality_project_cannot_be_called_directly():
    """Test that the project method cannot be called directly."""
    c_mat = jnp.array([[[1, 1]]])
    lb = jnp.zeros(shape=(1, 1, 1))
    ub = jnp.ones(shape=(1, 1, 1))
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "The 'project' method is not implemented and should not be called."
        ),
    ):
        ineq_constraint.project(ProjectionInstance(x=jnp.zeros((1, 2, 1))))
