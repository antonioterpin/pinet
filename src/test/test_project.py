"""Tests for the Project class."""

from itertools import product
from typing import cast

import cvxpy as cp
import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np
import pytest
from cvxpy.constraints.constraint import Constraint as CvxConstraint

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    BoxConstraintSpecification,
    EqualityConstraint,
    EqualityConstraintsSpecification,
    NonLinearConstraint,
    NonLinearSpecification,
    Project,
    ProjectionInstance,
    SOCType,
)

jax.config.update("jax_enable_x64", True)

SEEDS = [24, 42]
BATCH_SIZE = [1, 5]


# TODO: Add another test where var_a_mat, var_b are false.
@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZE))
def test_project_eq_ineq_var_a_dyn_varb(seed, batch_size):
    dim = 100
    n_eq = 40
    n_ineq = 50
    method = "pinv"
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, 10)
    # Generate equality constraint LHS
    a_mat = jax.random.normal(key[0], (batch_size, n_eq, dim))
    # Generate equality constraint RHS
    b = a_mat @ jax.random.normal(key[1], (batch_size, dim, 1))
    # Generate random point
    xinfeas = jax.random.normal(key[2], (batch_size, dim))
    # Compute projection with cvxpy
    yqp = jnp.zeros(shape=(batch_size, dim))
    for ii in range(batch_size):
        yprojcv = cp.Variable(dim)
        constraints = cast(
            list[CvxConstraint], [a_mat[ii, :, :] @ yprojcv == b[ii, :, 0]]
        )
        objective = cp.Minimize(cp.sum_squares(yprojcv - xinfeas[ii, :]))
        problem = cp.Problem(objective, constraints)
        problem.solve(verbose=False)
        yqp = yqp.at[ii, :].set(jnp.array(yprojcv.value).reshape(dim))

    # Compute projection with Project
    eq_constraint = EqualityConstraint(a_mat, b, method="pinv", var_b=True)
    projection_layer = Project(eq_constraint=eq_constraint)
    yprojiter = projection_layer.call(
        yraw=ProjectionInstance(
            x=xinfeas[..., None], eq=EqualityConstraintsSpecification(b=b)
        )
    )[0].x
    assert jnp.allclose(yprojiter[..., 0], yqp), (
        "Project should match the CVXPY equality-only solution. "
        f"Expected {yqp}, got {yprojiter[..., 0]}."
    )

    # Generate new RHS
    b_new = a_mat @ jax.random.normal(key[3], (batch_size, dim, 1))
    yprojiter = projection_layer.call(
        yraw=ProjectionInstance(
            x=xinfeas[..., None], eq=EqualityConstraintsSpecification(b=b_new)
        )
    )[0].x
    # New cvxpy problem
    yqp = jnp.zeros(shape=(batch_size, dim))
    for ii in range(batch_size):
        yprojcv = cp.Variable(dim)
        constraints_b_new = cast(
            list[CvxConstraint], [a_mat[ii, :, :] @ yprojcv == b_new[ii, :, 0]]
        )
        objective_b_new = cp.Minimize(cp.sum_squares(yprojcv - xinfeas[ii, :]))
        problem_b_new = cp.Problem(objective_b_new, constraints_b_new)
        problem_b_new.solve(verbose=False)
        yqp = yqp.at[ii, :].set(jnp.array(yprojcv.value).reshape(dim))

    assert jnp.allclose(yprojiter[..., 0], yqp), (
        "Project should recompute the equality-only solution after updating b. "
        f"Expected {yqp}, got {yprojiter[..., 0]}."
    )
    # %%
    # Generate inequality constraints LHS
    c_mat = jax.random.normal(key[4], shape=(batch_size, n_ineq, dim))
    b = jnp.zeros(shape=(batch_size, n_eq, 1))
    lb = jnp.zeros(shape=(batch_size, n_ineq, 1))
    ub = jnp.zeros(shape=(batch_size, n_ineq, 1))
    # Compute RHS by solving feasibility problem
    for ii in range(batch_size):
        xfeas = cp.Variable(dim)
        bfeas = cp.Variable(n_eq)
        lfeas = cp.Variable(n_ineq)
        ufeas = cp.Variable(n_ineq)
        constraints = cast(
            list[CvxConstraint],
            [
                a_mat[ii, :, :] @ xfeas == bfeas,
                lfeas <= c_mat[ii, :, :] @ xfeas,
                c_mat[ii, :, :] @ xfeas <= ufeas,
                -1 <= xfeas,
                xfeas <= 1,
            ],
        )
        objective = cp.Minimize(jnp.ones(shape=(dim)) @ xfeas)
        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve()

        # Extract RHS parameters
        b = b.at[ii, :, :].set(jnp.array(bfeas.value).reshape((n_eq, 1)))
        lb = lb.at[ii, :, :].set(jnp.array(lfeas.value).reshape((n_ineq, 1)))
        ub = ub.at[ii, :, :].set(jnp.array(ufeas.value).reshape((n_ineq, 1)))

    # Check projection layer without var_b
    eq_constraint = EqualityConstraint(a_mat=a_mat, b=b, method=method, var_b=False)
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)

    projection_layer_novarb = Project(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint
    )

    xprojiter_novarb = projection_layer_novarb.call(
        yraw=ProjectionInstance(x=xinfeas[..., None]),
        n_iter=500,
    )[0].x
    # Check projection layer with var_b
    eq_constraint = EqualityConstraint(a_mat=a_mat, b=b, method=method, var_b=True)
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)

    projection_layer = Project(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint
    )
    inp_varb = ProjectionInstance(
        x=xinfeas[..., None], eq=EqualityConstraintsSpecification(b=b)
    )
    xprojiter = projection_layer.call(yraw=inp_varb, n_iter=500)[0].x

    # Compute projections with QP
    yqp = jnp.zeros(shape=(batch_size, dim))
    for ii in range(batch_size):
        yproj = cp.Variable(dim)
        constraints = cast(
            list[CvxConstraint],
            [
                a_mat[ii, :, :] @ yproj == b[ii, :, 0],
                lb[ii, :, 0] <= c_mat[ii, :, :] @ yproj,
                c_mat[ii, :, :] @ yproj <= ub[ii, :, 0],
            ],
        )
        objective = cp.Minimize(cp.sum_squares(yproj - xinfeas[ii, :]))
        problem_qp = cp.Problem(objective=objective, constraints=constraints)
        problem_qp.solve()
        yqp = yqp.at[ii, :].set(jnp.array(yproj.value).reshape(dim))

    assert jnp.allclose(xprojiter[..., 0], yqp, atol=1e-3, rtol=1e-3), (
        "Project with variable b should match the equality-plus-inequality QP "
        f"solution. Expected {yqp}, got {xprojiter[..., 0]}."
    )
    assert jnp.allclose(xprojiter_novarb[..., 0], yqp, atol=1e-3, rtol=1e-3), (
        "Project with fixed b should match the equality-plus-inequality QP "
        f"solution. Expected {yqp}, got {xprojiter_novarb[..., 0]}."
    )
    # Test call and check method
    sigma = 1.0
    omega = 1.7
    tol = 1e-5
    check_every = 20
    max_iter = 500
    reduction = "max"
    for reduction in ["max", "mean", 0.5, 0.9]:
        check = projection_layer.call_and_check(
            sigma=sigma,
            omega=omega,
            check_every=check_every,
            tol=tol,
            max_iter=max_iter,
            reduction=reduction,
        )
        check_novarb = projection_layer_novarb.call_and_check(
            sigma=sigma,
            omega=omega,
            check_every=check_every,
            tol=tol,
            max_iter=max_iter,
            reduction=reduction,
        )
        _, flag, _ = check(
            ProjectionInstance(
                x=xinfeas[..., None], eq=EqualityConstraintsSpecification(b=b)
            )
        )
        _, flag_novarb, _ = check_novarb(ProjectionInstance(x=xinfeas[..., None]))

        assert flag, (
            "call_and_check should report convergence for the variable-b "
            f"projection with reduction={reduction}."
        )
        assert flag_novarb, (
            "call_and_check should report convergence for the fixed-b "
            f"projection with reduction={reduction}."
        )

    # %%
    b_new = b + jax.random.normal(key[5], shape=(batch_size, n_eq, 1))
    yqp = jnp.zeros(shape=(batch_size, dim))
    for ii in range(batch_size):
        yproj = cp.Variable(dim)
        constraints = cast(
            list[CvxConstraint],
            [
                a_mat[ii, :, :] @ yproj == b_new[ii, :, 0],
                lb[ii, :, 0] <= c_mat[ii, :, :] @ yproj,
                c_mat[ii, :, :] @ yproj <= ub[ii, :, 0],
            ],
        )
        objective = cp.Minimize(cp.sum_squares(yproj - xinfeas[ii, :]))
        problem_qp = cp.Problem(objective=objective, constraints=constraints)
        problem_qp.solve(verbose=False)
        yqp = yqp.at[ii, :].set(jnp.array(yproj.value).reshape(dim))

    assert inp_varb.eq is not None, (
        "Projection input for the variable-b case should contain equality data."
    )
    inp_varb_new = inp_varb.update(eq=inp_varb.eq.update(b=b_new))
    xprojiter = projection_layer.call(yraw=inp_varb_new, n_iter=500)[0].x
    assert jnp.allclose(xprojiter[..., 0], yqp, atol=1e-3, rtol=1e-3), (
        "Project should match the QP solution after updating the equality RHS. "
        f"Expected {yqp}, got {xprojiter[..., 0]}."
    )
    # %%
    # Generate new LHS and RHS
    a_dyn_new = jax.random.normal(key[6], (batch_size, n_eq, dim))
    b_new = a_dyn_new @ jax.random.normal(key[7], (batch_size, dim, 1))
    eq_constraint = EqualityConstraint(
        a_mat=a_dyn_new, b=b_new, method=method, var_a_mat=True
    )
    projection_layer = Project(eq_constraint=eq_constraint)
    inp = ProjectionInstance(
        x=xinfeas[..., None],
        eq=EqualityConstraintsSpecification(a_mat=a_dyn_new, b=b_new),
    )
    xprojiter = projection_layer.call(yraw=inp)[0].x
    # New cvxpy problem
    yqp = jnp.zeros(shape=(batch_size, dim))
    for ii in range(batch_size):
        yprojcv = cp.Variable(dim)
        constraints_new = cast(
            list[CvxConstraint], [a_dyn_new[ii, :, :] @ yprojcv == b_new[ii, :, 0]]
        )
        objective_new = cp.Minimize(cp.sum_squares(yprojcv - xinfeas[ii, :]))
        problem_new = cp.Problem(objective_new, constraints_new)
        problem_new.solve(verbose=False)
        yqp = yqp.at[ii, :].set(jnp.array(yprojcv.value).reshape(dim))

    assert jnp.allclose(xprojiter[..., 0], yqp), (
        "Project should match the CVXPY solution after updating the equality "
        f"matrix and RHS. Expected {yqp}, got {xprojiter[..., 0]}."
    )
    # %% Solve projection with both equality and inequality
    yqp = jnp.zeros(shape=(batch_size, dim))
    for ii in range(batch_size):
        yproj = cp.Variable(dim)
        constraints = cast(
            list[CvxConstraint],
            [
                a_dyn_new[ii, :, :] @ yproj == b_new[ii, :, 0],
                lb[ii, :, 0] <= c_mat[ii, :, :] @ yproj,
                c_mat[ii, :, :] @ yproj <= ub[ii, :, 0],
            ],
        )
        objective = cp.Minimize(cp.sum_squares(yproj - xinfeas[ii, :]))
        problem_qp = cp.Problem(objective=objective, constraints=constraints)
        problem_qp.solve(verbose=False)
        yqp = yqp.at[ii, :].set(jnp.array(yproj.value).reshape(dim))

    eq_constraint = EqualityConstraint(
        a_mat=a_mat, b=b, method=method, var_b=True, var_a_mat=True
    )
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)

    projection_layer = Project(
        eq_constraint=eq_constraint, ineq_constraint=ineq_constraint
    )
    inp = ProjectionInstance(
        x=xinfeas[..., None],
        eq=EqualityConstraintsSpecification(b=b_new, a_mat=a_dyn_new),
    )
    xprojiter = projection_layer.call(yraw=inp, n_iter=500)[0].x

    assert jnp.allclose(xprojiter.reshape(yqp.shape), yqp, atol=1e-3, rtol=1e-3), (
        "Project should match the QP solution when both equality and inequality "
        f"data are variable. Expected {yqp}, got {xprojiter.reshape(yqp.shape)}."
    )


def test_call_default_n_iter_projects_correctly():
    """Regression test for PR #94: default n_iter was 0,
       causing call() to return yraw unchanged.

    With n_iter=0 the scan was skipped and the raw input was returned without
    projecting. The fix sets the default to 100 and adds an assertion, so calling call()
    without an explicit n_iter must now produce a valid projection.
    """
    dim, n_eq, batch = 20, 5, 1
    key = jax.random.PRNGKey(7)
    ka_mat, kx0, kx = jax.random.split(key, 3)

    a_mat = jax.random.normal(ka_mat, (batch, n_eq, dim))
    x0 = jax.random.normal(kx0, (batch, dim, 1))
    b = a_mat @ x0  # feasible RHS constructed from a random point

    # Deliberately pick a point that does NOT satisfy A x = b
    xinfeas = jax.random.normal(kx, (batch, dim, 1))
    assert jnp.linalg.norm(a_mat @ xinfeas - b) > 1e-3, "Test point must be infeasible"

    eq = EqualityConstraint(a_mat=a_mat, b=b, method="pinv", var_b=False)
    layer = Project(eq_constraint=eq)

    # Call without specifying n_iter — uses the default (100 after the fix, 0 before)
    result = layer.call(yraw=ProjectionInstance(x=xinfeas))[0].x

    # The projected point must satisfy the equality constraint
    residual = jnp.linalg.norm(a_mat @ result - b)
    assert residual < 1e-4, f"Expected ||Ax - b|| < 1e-4, got {residual:.6f}"

    assert not jnp.allclose(result, xinfeas), (
        "Projection must differ from the infeasible input (n_iter=0 regression)"
    )


def test_call_n_iter_zero_raises():
    """Regression test for PR #94: n_iter=0 must raise AssertionError after the fix.

    Before the fix, passing n_iter=0 silently returned the raw input. The fix replaces the
    conditional branch with an explicit assertion so that invalid inputs are caught early.
    """
    dim, n_eq, batch = 5, 2, 1
    key = jax.random.PRNGKey(0)
    ka_mat, kx0, kx = jax.random.split(key, 3)

    a_mat = jax.random.normal(ka_mat, (batch, n_eq, dim))
    x0 = jax.random.normal(kx0, (batch, dim, 1))
    b = a_mat @ x0

    eq = EqualityConstraint(a_mat=a_mat, b=b, method="pinv", var_b=False)
    layer = Project(eq_constraint=eq)

    xinfeas = jax.random.normal(kx, (batch, dim, 1))

    with pytest.raises(AssertionError, match=r"Number of iterations must be positive"):
        layer.call(yraw=ProjectionInstance(x=xinfeas), n_iter=0)


@pytest.mark.parametrize("bad_reduction", ["median", 1.5, 0.0, -0.2, 2])
def test_call_and_check_invalid_reduction_raises(bad_reduction):
    # Minimal feasible setup: a_mat x = b with b constructed from a random x0
    dim, n_eq, batch = 5, 2, 1
    key = jax.random.PRNGKey(0)
    k_a_dyn, kx0, kx = jax.random.split(key, 3)

    a_mat = jax.random.normal(k_a_dyn, (batch, n_eq, dim))
    x0 = jax.random.normal(kx0, (batch, dim, 1))
    b = a_mat @ x0

    eq = EqualityConstraint(a_mat=a_mat, b=b, method="pinv", var_b=False)
    layer = Project(eq_constraint=eq)

    xinfeas = jax.random.normal(kx, (batch, dim, 1))
    project_and_check = layer.call_and_check(
        reduction=bad_reduction, check_every=1, max_iter=1
    )

    with pytest.raises(ValueError, match=r"Invalid reduction method"):
        project_and_check(ProjectionInstance(x=xinfeas))


@pytest.mark.parametrize("seed", SEEDS)
def test_project_cv_linear_constraints(seed):
    """Test Project.cv for linear constraints (eq + ineq, no nonlinear)."""
    dim = 4
    n_eq = 1
    n_ineq = 2
    key = jrnd.PRNGKey(seed)
    key, ka_mat, kx, kc_mat = jrnd.split(key, 4)

    a_mat = jrnd.normal(ka_mat, shape=(1, n_eq, dim))
    x_feas = jrnd.uniform(kx, shape=(1, dim, 1), minval=-1.0, maxval=1.0)
    b = a_mat @ x_feas
    eq_constraint = EqualityConstraint(a_mat=a_mat, b=b, var_b=False)

    c_mat = jrnd.normal(kc_mat, shape=(1, n_ineq, dim))
    Cx = c_mat @ x_feas
    lb = Cx - 0.1
    ub = Cx + 0.1
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)

    layer = Project(
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        box_constraint=None,
    )

    xinfeas = jnp.ones((1, dim, 1)) * 3.0
    yraw = ProjectionInstance(x=xinfeas)

    cv_layer = layer.cv(yraw)
    y_lifted = layer.lift(yraw)
    # The non-simple polytope path always populates these lifted attributes.
    assert layer.lifted_eq_constraint is not None
    assert layer.lifted_primitive_constraint is not None
    cv_expected = jnp.maximum(
        layer.lifted_eq_constraint.cv(y_lifted),
        layer.lifted_primitive_constraint.cv(y_lifted),
    )

    assert jnp.allclose(cv_layer, cv_expected)


@pytest.mark.parametrize("seed", SEEDS)
def test_project_cv_nonlinear_constraints(seed):
    """Test Project.cv for nonlinear constraints (eq + ineq + SOC, no box)."""
    dim = 3
    n_eq = 1
    n_ineq = 1
    key = jrnd.PRNGKey(seed)
    key, ka_mat, kx, kc_mat, knA, knf = jrnd.split(key, 6)

    a_mat = jrnd.normal(ka_mat, shape=(1, n_eq, dim))
    x_feas = jrnd.uniform(kx, shape=(1, dim, 1), minval=-0.5, maxval=0.5)
    b = a_mat @ x_feas
    eq_constraint = EqualityConstraint(a_mat=a_mat, b=b, var_b=False)

    c_mat = jrnd.normal(kc_mat, shape=(1, n_ineq, dim))
    Cx = c_mat @ x_feas
    lb = Cx - 0.2
    ub = Cx + 0.2
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)

    a_soc_mat = jrnd.normal(knA, shape=(1, 2, dim))
    a_soc = jnp.zeros((1, 2, 1))
    f_soc = jrnd.normal(knf, shape=(1, 1, dim))
    # Build b so that x_feas is safely feasible for the SOC constraint.
    b_soc = (
        jnp.linalg.norm(a_soc_mat @ x_feas + a_soc, ord=2, axis=1, keepdims=True)
        - f_soc @ x_feas
        + 0.5
    )
    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=a_soc_mat,
        a=a_soc,
        f=f_soc,
        b=b_soc,
    )
    nl_constraint = NonLinearConstraint(spec=nl_spec)

    layer = Project(
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        box_constraint=None,
        nl_constraints=[nl_constraint],
    )

    xinfeas = jnp.ones((1, dim, 1)) * 2.0
    yraw = ProjectionInstance(x=xinfeas, nl=[nl_spec])

    cv_layer = layer.cv(yraw)
    y_lifted = layer.lift(yraw)
    # The non-linear path always populates these lifted attributes.
    assert layer.lifted_eq_constraint is not None
    assert layer.lifted_primitive_constraint is not None
    cv_expected = jnp.maximum(
        layer.lifted_eq_constraint.cv(y_lifted),
        layer.lifted_primitive_constraint.cv(y_lifted),
    )

    assert jnp.allclose(cv_layer, cv_expected)


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZE))
def test_project_box_ineq_eq_soc(seed, batch_size):
    dim = 200
    n_a = 12
    n_c = 25
    n_a_soc_1 = 32
    n_a_soc_2 = 42
    key = jrnd.PRNGKey(seed)
    # Generate a random point which will be feasible by construction
    key, subkey = jrnd.split(key)
    x_feas = jrnd.uniform(subkey, shape=(1, dim, 1), minval=-2, maxval=2)

    # Equality constraint
    a_mat = jrnd.uniform(key, shape=(1, n_a, dim), minval=-2, maxval=2)
    b = a_mat @ x_feas
    eq_constraint = EqualityConstraint(a_mat=a_mat, b=b, var_b=False)

    # Box constraint
    mask = jnp.array([True] * dim, dtype=jnp.bool_)
    lb_box = jnp.array([-2.0] * dim).reshape(1, -1, 1)
    ub_box = jnp.array([2.0] * dim).reshape(1, -1, 1)
    box_spec = BoxConstraintSpecification(mask=mask, lb=lb_box, ub=ub_box)
    box_spec.validate()
    box_constraint = BoxConstraint(box_spec=box_spec)

    # Inequality constraint
    eps_ineq = 1e-2  # slack for inequality constraints
    key, subkey = jrnd.split(key)
    c_mat = jrnd.uniform(subkey, shape=(1, n_c, dim), minval=-2, maxval=2)
    lb_ineq = c_mat @ x_feas - eps_ineq
    key, subkey = jrnd.split(key)
    ub_ineq = lb_ineq + jrnd.uniform(subkey, shape=(1, n_c, 1), minval=0, maxval=1)
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb_ineq, ub=ub_ineq)

    # SOC constraint 1
    eps_soc = 1e-2  # Slack to ensure feasibility of x_feas
    key, subkey = jrnd.split(key)
    a_soc_1_mat = jrnd.uniform(subkey, shape=(1, n_a_soc_1, dim), minval=-2, maxval=2)
    key, subkey = jrnd.split(key)
    a_soc_1 = jrnd.uniform(subkey, shape=(batch_size, n_a_soc_1, 1), minval=0.5, maxval=2)
    key, subkey = jrnd.split(key)
    f_soc_1 = jrnd.uniform(subkey, shape=(1, 1, dim), minval=0, maxval=1)
    b_soc_1 = (
        eps_soc
        + jnp.linalg.norm(a_soc_1_mat @ x_feas + a_soc_1, ord=2, axis=1, keepdims=True)
        - f_soc_1 @ x_feas
    )

    nl_spec_1 = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=a_soc_1_mat,
        a=a_soc_1,
        f=f_soc_1,
        b=b_soc_1,
    )
    soc_constraint_1 = NonLinearConstraint(
        spec=nl_spec_1,
    )

    # SOC constraint 2
    key, subkey = jrnd.split(key)
    a_soc_2_mat = jrnd.uniform(subkey, shape=(1, n_a_soc_2, dim), minval=-2, maxval=2)
    key, subkey = jrnd.split(key)
    a_soc_2 = jrnd.uniform(subkey, shape=(batch_size, n_a_soc_2, 1), minval=0.5, maxval=2)
    key, subkey = jrnd.split(key)
    f_soc_2 = jrnd.uniform(subkey, shape=(1, 1, dim), minval=-1, maxval=1)
    b_soc_2 = (
        eps_soc
        + jnp.linalg.norm(a_soc_2_mat @ x_feas + a_soc_2, ord=2, axis=1, keepdims=True)
        - f_soc_2 @ x_feas
    )
    nl_spec_2 = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=a_soc_2_mat,
        a=a_soc_2,
        f=f_soc_2,
        b=b_soc_2,
    )
    soc_constraint_2 = NonLinearConstraint(
        spec=nl_spec_2,
    )

    nl_constraints = [
        soc_constraint_1,
        soc_constraint_2,
    ]
    # Build projection layer
    projection_layer = Project(
        eq_constraint=eq_constraint,
        box_constraint=box_constraint,
        ineq_constraint=ineq_constraint,
        nl_constraints=nl_constraints,
    )
    # Generate points to be projected
    key, subkey = jrnd.split(key)
    yproj = jrnd.uniform(subkey, shape=(batch_size, dim, 1), minval=-5, maxval=5)
    yraw = ProjectionInstance(x=yproj, nl=[nl_spec_1, nl_spec_2])

    # Run projection
    n_iter = 5000
    sigma = 5.0
    omega = 1.7
    yk, sk = projection_layer.call(yraw=yraw, n_iter=n_iter, sigma=sigma, omega=omega)

    # Compute projection with cvxpy
    y_cvxpy = cp.Variable(dim)
    x_cvxpy = cp.Parameter(dim)
    b_eq_cvxpy = cp.Parameter(n_a)
    a_soc_1_cvxpy = cp.Parameter(n_a_soc_1)
    b_soc_1_cvxpy = cp.Parameter(1)
    a_soc_2_cvxpy = cp.Parameter(n_a_soc_2)
    b_soc_2_cvxpy = cp.Parameter(1)
    constraints = [
        a_mat[0, :, :] @ y_cvxpy == b_eq_cvxpy,
        lb_box[0, :, 0] <= y_cvxpy[mask],
        y_cvxpy[mask] <= ub_box[0, :, 0],
        lb_ineq[0, :, 0] <= c_mat[0, :, :] @ y_cvxpy,
        c_mat[0, :, :] @ y_cvxpy <= ub_ineq[0, :, 0],
        cp.SOC(
            f_soc_1[0, :, :] @ y_cvxpy + b_soc_1_cvxpy,
            a_soc_1_mat[0, :, :] @ y_cvxpy + a_soc_1_cvxpy,
        ),
        cp.SOC(
            f_soc_2[0, :, :] @ y_cvxpy + b_soc_2_cvxpy,
            a_soc_2_mat[0, :, :] @ y_cvxpy + a_soc_2_cvxpy,
        ),
    ]
    objective = cp.Minimize(cp.sum_squares(y_cvxpy - x_cvxpy))
    problem_cvxpy = cp.Problem(
        objective=objective, constraints=cast(list[CvxConstraint], constraints)
    )
    y_opt = jnp.zeros((batch_size, dim, 1))
    for ii in range(batch_size):
        x_cvxpy.value = np.array(yproj[ii].reshape(-1))
        b_eq_cvxpy.value = np.array(b[0, :, 0])
        a_soc_1_cvxpy.value = np.array(a_soc_1[ii, :, 0])
        b_soc_1_cvxpy.value = np.array(b_soc_1[ii, :, 0])
        a_soc_2_cvxpy.value = np.array(a_soc_2[ii, :, 0])
        b_soc_2_cvxpy.value = np.array(b_soc_2[ii, :, 0])
        problem_cvxpy.solve(solver=cp.SCS, verbose=False, eps_abs=1e-10, eps_rel=1e-10)
        y_opt = y_opt.at[ii].set(jnp.array(y_cvxpy.value).reshape(-1, 1))

    assert jnp.allclose(yk.x, y_opt, atol=1e-6, rtol=1e-6), """
        Projected points do not match CVXPY solution.
    """
    # The non-linear path populates the lifted equality constraint.
    assert projection_layer.lifted_eq_constraint is not None
    assert jnp.allclose(
        projection_layer.step_final(sk).x[:, dim:, :],
        projection_layer.lifted_eq_constraint.a_mat[0, n_a:, :dim] @ y_opt,
        atol=1e-5,
        rtol=1e-5,
    ), """
        Auxiliary variables do not match CVXPY solution.
    """

    # Generate new soc constraint parameters
    key, subkey = jrnd.split(key)
    a_soc_1_new = jrnd.uniform(
        subkey, shape=(batch_size, n_a_soc_1, 1), minval=0.5, maxval=2
    )
    b_soc_1_new = (
        eps_soc
        + jnp.linalg.norm(
            a_soc_1_mat @ x_feas + a_soc_1_new, ord=2, axis=1, keepdims=True
        )
        - f_soc_1 @ x_feas
    )
    nl_spec_1_new = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=a_soc_1_mat,
        a=a_soc_1_new,
        f=f_soc_1,
        b=b_soc_1_new,
    )

    key, subkey = jrnd.split(key)
    a_soc_2_new = jrnd.uniform(
        subkey, shape=(batch_size, n_a_soc_2, 1), minval=0.5, maxval=2
    )
    b_soc_2_new = (
        eps_soc
        + jnp.linalg.norm(
            a_soc_2_mat @ x_feas + a_soc_2_new, ord=2, axis=1, keepdims=True
        )
        - f_soc_2 @ x_feas
    )
    nl_spec_2_new = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=a_soc_2_mat,
        a=a_soc_2_new,
        f=f_soc_2,
        b=b_soc_2_new,
    )
    yraw_new = yraw.update(nl=[nl_spec_1_new, nl_spec_2_new])
    yk_new, sk_new = projection_layer.call(
        yraw=yraw_new, n_iter=n_iter, sigma=sigma, omega=omega
    )

    y_opt_new = jnp.zeros((batch_size, dim, 1))
    for ii in range(batch_size):
        x_cvxpy.value = np.array(yproj[ii].reshape(-1))
        b_eq_cvxpy.value = np.array(b[0, :, 0])
        a_soc_1_cvxpy.value = np.array(a_soc_1_new[ii, :, 0])
        b_soc_1_cvxpy.value = np.array(b_soc_1_new[ii, :, 0])
        a_soc_2_cvxpy.value = np.array(a_soc_2_new[ii, :, 0])
        b_soc_2_cvxpy.value = np.array(b_soc_2_new[ii, :, 0])
        problem_cvxpy.solve(solver=cp.SCS, verbose=False, eps_abs=1e-10, eps_rel=1e-10)
        y_opt_new = y_opt_new.at[ii].set(jnp.array(y_cvxpy.value).reshape(-1, 1))

    assert jnp.allclose(yk_new.x, y_opt_new, atol=1e-6, rtol=1e-6), """
        Projected points do not match CVXPY solution.
    """
    # The non-linear path populates the lifted equality constraint.
    assert projection_layer.lifted_eq_constraint is not None
    assert jnp.allclose(
        projection_layer.step_final(sk_new).x[:, dim:, :],
        projection_layer.lifted_eq_constraint.a_mat[0, n_a:, :dim] @ y_opt_new,
        atol=1e-5,
        rtol=1e-5,
    ), """
        Auxiliary variables do not match CVXPY solution.
    """
