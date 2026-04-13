from itertools import product

import cvxpy as cp
import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np
import pytest

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    BoxConstraintSpecification,
    ConstraintParser,
    EqualityConstraint,
    NonLinearConstraint,
    NonLinearSpecification,
    ProjectionInstance,
    SocConstraint,
    SOCType,
    build_iteration_step,
)
from pinet.constraints.non_linear_types import L2NormType

jax.config.update("jax_enable_x64", True)
SEEDS = [0, 24, 42]
BATCH_SIZES = [1, 10, 50]


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZES))
def test_simple_problem(seed, batch_size):
    dim = 2
    n_A = 1
    n_C = 2
    n_A_soc_1 = 3
    n_A_soc_2 = 4
    key = jrnd.PRNGKey(seed)
    # Generate a random point which will be feasible by-construction
    key, subkey = jrnd.split(key)
    x_feas = jrnd.uniform(subkey, shape=(1, dim, 1), minval=-2, maxval=2)

    # Equality constraint
    A = jrnd.uniform(key, shape=(1, n_A, dim), minval=-2, maxval=2)
    b = A @ x_feas
    eq_constraint = EqualityConstraint(A=A, b=b, var_b=False)

    # Box constraint
    mask = jnp.array([True] + [False] * (dim - 1), dtype=jnp.bool_)
    lb_box = jnp.array([-2.0]).reshape(1, -1, 1)
    ub_box = jnp.array([2.0]).reshape(1, -1, 1)
    box_spec = BoxConstraintSpecification(mask=mask, lb=lb_box, ub=ub_box)
    box_spec.validate()
    box_constraint = BoxConstraint(box_spec=box_spec)

    # Inequality constraint
    eps_ineq = 1e-2  # slack for inequality constraints
    key, subkey = jrnd.split(key)
    C = jrnd.uniform(subkey, shape=(1, n_C, dim), minval=-2, maxval=2)
    lb_ineq = C @ x_feas - eps_ineq
    key, subkey = jrnd.split(key)
    ub_ineq = lb_ineq + jrnd.uniform(subkey, shape=(1, n_C, 1), minval=0, maxval=1)
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb_ineq, ub=ub_ineq)

    # SOC constraint 1
    eps_soc = 1e-2  # Slack to ensure feasibility of x_feas
    key, subkey = jrnd.split(key)
    A_soc_1 = jrnd.uniform(subkey, shape=(1, n_A_soc_1, dim), minval=-2, maxval=2)
    key, subkey = jrnd.split(key)
    a_soc_1 = jrnd.uniform(subkey, shape=(1, n_A_soc_1, 1), minval=0.5, maxval=2)
    key, subkey = jrnd.split(key)
    f_soc_1 = jrnd.uniform(subkey, shape=(1, 1, dim), minval=0, maxval=1)
    b_soc_1 = (
        eps_soc
        + jnp.linalg.norm(A_soc_1 @ x_feas + a_soc_1, ord=2, axis=1)
        - f_soc_1 @ x_feas
    )
    nlspec_1 = NonLinearSpecification(
        A=A_soc_1,
        a=a_soc_1,
        f=f_soc_1,
        b=b_soc_1,
        nl_type=SOCType,
    )
    soc_constraint_1 = NonLinearConstraint(
        spec=nlspec_1,
    )

    # SOC constraint 2
    key, subkey = jrnd.split(key)
    A_soc_2 = jrnd.uniform(subkey, shape=(1, n_A_soc_2, dim), minval=-2, maxval=2)
    key, subkey = jrnd.split(key)
    a_soc_2 = jrnd.uniform(subkey, shape=(1, n_A_soc_2, 1), minval=0.5, maxval=2)
    key, subkey = jrnd.split(key)
    f_soc_2 = jrnd.uniform(subkey, shape=(1, 1, dim), minval=-1, maxval=1)
    b_soc_2 = (
        eps_soc
        + jnp.linalg.norm(A_soc_2 @ x_feas + a_soc_2, ord=2, axis=1)
        - f_soc_2 @ x_feas
    )
    nlspec_2 = NonLinearSpecification(
        A=A_soc_2,
        a=a_soc_2,
        f=f_soc_2,
        b=b_soc_2,
        nl_type=SOCType,
    )
    soc_constraint_2 = NonLinearConstraint(
        spec=nlspec_2,
    )
    # Parse constraints
    nl_constraints = [
        soc_constraint_1,
        soc_constraint_2,
    ]
    parser = ConstraintParser(
        eq_constraint=eq_constraint,
        box_constraint=box_constraint,
        ineq_constraint=ineq_constraint,
        nl_constraints=nl_constraints,
    )
    (eq_lifted, cart_lifted, _) = parser.parse()

    # Check parsing
    # Dimension of augmented space
    # n_aug = dim + n_C + n_A_soc_1 + 1 + n_A_soc_2 + 1
    # Extra dimensions in augmented
    n_extra = n_C + n_A_soc_1 + 1 + n_A_soc_2 + 1
    A_lifted_correct = jnp.concatenate(
        [
            jnp.concatenate([A, C, A_soc_1, f_soc_1, A_soc_2, f_soc_2], axis=1),
            jnp.zeros((1, n_A + n_extra, n_extra)),
        ],
        axis=2,
    )
    A_lifted_correct = A_lifted_correct + jnp.zeros_like(A_lifted_correct).at[
        :, n_A:, dim:
    ].set(-jnp.eye(n_extra))
    b_lifted_correct = jnp.concatenate([b, jnp.zeros((1, n_extra, 1))], axis=1)
    box_mask_correct = jnp.concatenate(
        [mask, jnp.array([True] * n_C + [False] * (n_extra - n_C), dtype=jnp.bool_)]
    )
    box_ub_correct = jnp.concatenate([ub_box, ub_ineq], axis=1)
    box_lb_correct = jnp.concatenate([lb_box, lb_ineq], axis=1)
    soc_1_mask_u_correct = jnp.array(
        [False] * (dim + n_C) + [True] * n_A_soc_1 + [False] * (1 + n_A_soc_2 + 1),
        dtype=jnp.bool_,
    )
    soc_1_mask_t_correct = jnp.array(
        [False] * (dim + n_C + n_A_soc_1) + [True] + [False] * (n_A_soc_2 + 1),
        dtype=jnp.bool_,
    )
    soc_2_mask_u_correct = jnp.array(
        [False] * (dim + n_C + n_A_soc_1 + 1) + [True] * n_A_soc_2 + [False],
        dtype=jnp.bool_,
    )
    soc_2_mask_t_correct = jnp.array(
        [False] * (dim + n_C + n_A_soc_1 + 1 + n_A_soc_2) + [True], dtype=jnp.bool_
    )

    # Assertions
    assert jnp.allclose(
        eq_lifted.A, A_lifted_correct
    ), """Lifted A matrix is incorrect."""
    assert jnp.allclose(
        eq_lifted.b, b_lifted_correct
    ), """
        Lifted b vector is incorrect.
    """
    assert isinstance(
        cart_lifted.constraints[0], BoxConstraint
    ), """
        First constraint should be BoxConstraint.
    """
    assert jnp.allclose(
        cart_lifted.constraints[0].mask, box_mask_correct
    ), """
        Box mask is incorrect.
    """
    assert jnp.allclose(
        cart_lifted.constraints[0].ub, box_ub_correct
    ), """
        Box upper bound is incorrect.
    """
    assert jnp.allclose(
        cart_lifted.constraints[0].lb, box_lb_correct
    ), """
        Box lower bound is incorrect.
    """
    assert isinstance(
        cart_lifted.constraints[1], SocConstraint
    ), """
        Second constraint should be SocConstraint.
    """
    assert jnp.allclose(
        cart_lifted.constraints[1].mask_u, soc_1_mask_u_correct
    ), """
        SOC 1 mask_u is incorrect.
    """
    assert jnp.allclose(
        cart_lifted.constraints[1].mask_t, soc_1_mask_t_correct
    ), """
        SOC 1 mask_t is incorrect.
    """
    assert isinstance(
        cart_lifted.constraints[2], SocConstraint
    ), """
        Third constraint should be SocConstraint.
    """
    assert jnp.allclose(
        cart_lifted.constraints[2].mask_u, soc_2_mask_u_correct
    ), """
        SOC 2 mask_u is incorrect.
    """
    assert jnp.allclose(
        cart_lifted.constraints[2].mask_t, soc_2_mask_t_correct
    ), """
        SOC 2 mask_t is incorrect.
    """

    # Create random points to be projected
    key, subkey = jrnd.split(key)
    yproj = jrnd.uniform(subkey, shape=(batch_size, dim, 1), minval=-5, maxval=5)
    yraw = ProjectionInstance(x=yproj, nl=[nlspec_1, nlspec_2])

    # Build the algorithm
    n_iter = 1500
    iteration_step, final_step = build_iteration_step(
        eq_constraint=eq_lifted,
        box_constraint=cart_lifted,
        dim=dim,
    )
    iteration_step = jax.jit(iteration_step)
    sk = ProjectionInstance(
        x=jnp.zeros((batch_size, dim + n_extra, 1)), nl=[nlspec_1, nlspec_2]
    )
    for ii in range(n_iter):
        sk = iteration_step(sk=sk, yraw=yraw, sigma=0.1, omega=1.8)
    yk = final_step(sk)

    # Compute projection with cvxpy
    y_cvxpy = cp.Variable(dim)
    x_cvxpy = cp.Parameter(dim)
    constraints = [
        A[0, :, :] @ y_cvxpy == b[0, :, 0],
        lb_box[0, :, 0] <= y_cvxpy[mask],
        y_cvxpy[mask] <= ub_box[0, :, 0],
        lb_ineq[0, :, 0] <= C[0, :, :] @ y_cvxpy,
        C[0, :, :] @ y_cvxpy <= ub_ineq[0, :, 0],
        cp.SOC(
            f_soc_1[0, :, :] @ y_cvxpy + b_soc_1[0, :, 0],
            A_soc_1[0, :, :] @ y_cvxpy + a_soc_1[0, :, 0],
        ),
        cp.SOC(
            f_soc_2[0, :, :] @ y_cvxpy + b_soc_2[0, :, 0],
            A_soc_2[0, :, :] @ y_cvxpy + a_soc_2[0, :, 0],
        ),
    ]
    objective = cp.Minimize(cp.sum_squares(y_cvxpy - x_cvxpy))
    problem_cvxpy = cp.Problem(objective=objective, constraints=constraints)
    y_opt = jnp.zeros((batch_size, dim, 1))
    for ii in range(batch_size):
        x_cvxpy.value = np.array(yproj[ii].reshape(-1))
        problem_cvxpy.solve(solver=cp.SCS, verbose=False, eps_abs=1e-9, eps_rel=1e-9)
        y_opt = y_opt.at[ii].set(jnp.array(y_cvxpy.value).reshape(-1, 1))

    assert jnp.allclose(
        yk.x[:, :dim, :], y_opt, atol=1e-5, rtol=1e-5
    ), """
        Projected points do not match CVXPY solution.
    """
    assert jnp.allclose(
        yk.x[:, dim:, :], eq_lifted.A[0, n_A:, :dim] @ y_opt, atol=1e-5, rtol=1e-5
    ), """
        Auxiliary variables do not match CVXPY solution.
    """

    # Resolve with different values for a_soc_2 and b_soc_2
    key, subkey = jrnd.split(key)
    a_soc_2_new = jrnd.uniform(subkey, shape=(1, n_A_soc_2, 1), minval=0.5, maxval=2)
    key, subkey = jrnd.split(key)
    b_soc_2_new = (
        eps_soc
        + jnp.linalg.norm(A_soc_2 @ x_feas + a_soc_2_new, ord=2, axis=1)
        - f_soc_2 @ x_feas
    )
    nlspec_2_new = nlspec_2.update(a=a_soc_2_new, b=b_soc_2_new)
    yraw_new = ProjectionInstance(x=yproj, nl=[nlspec_1, nlspec_2_new])
    sk = ProjectionInstance(
        x=jnp.zeros((batch_size, dim + n_extra, 1)), nl=[nlspec_1, nlspec_2_new]
    )
    for ii in range(n_iter):
        sk = iteration_step(sk=sk, yraw=yraw_new, sigma=0.1, omega=1.8)
    yk_new = final_step(sk)

    # CVXPY solution with updated SOC parameters
    constraints[6] = cp.SOC(
        f_soc_2[0, :, :] @ y_cvxpy + b_soc_2_new[0, :, 0],
        A_soc_2[0, :, :] @ y_cvxpy + a_soc_2_new[0, :, 0],
    )
    problem_cvxpy = cp.Problem(objective=objective, constraints=constraints)
    y_opt_new = jnp.zeros((batch_size, dim, 1))
    for ii in range(batch_size):
        x_cvxpy.value = np.array(yproj[ii].reshape(-1))
        problem_cvxpy.solve(solver=cp.SCS, verbose=False, eps_abs=1e-9, eps_rel=1e-9)
        y_opt_new = y_opt_new.at[ii].set(jnp.array(y_cvxpy.value).reshape(-1, 1))

    assert jnp.allclose(
        yk_new.x[:, :dim, :], y_opt_new, atol=1e-5, rtol=1e-5
    ), """
        Projected points do not match CVXPY solution.
    """
    assert jnp.allclose(
        yk_new.x[:, dim:, :],
        eq_lifted.A[0, n_A:, :dim] @ y_opt_new,
        atol=1e-5,
        rtol=1e-5,
    ), """
        Auxiliary variables do not match CVXPY solution.
    """


def test_parse_non_linear_with_no_box_constraint():
    """Test parser behavior when box constraint is None."""
    dim = 3
    n_eq = 1
    n_ineq = 2
    key = jrnd.PRNGKey(0)

    key, ka, kc, ks, kf, ka_soc = jrnd.split(key, 6)
    A = jrnd.uniform(ka, shape=(1, n_eq, dim), minval=-1, maxval=1)
    x_ref = jrnd.uniform(ks, shape=(1, dim, 1), minval=-1, maxval=1)
    b = A @ x_ref
    eq_constraint = EqualityConstraint(A=A, b=b, var_b=False)

    C = jrnd.uniform(kc, shape=(1, n_ineq, dim), minval=-1, maxval=1)
    Cx = C @ x_ref
    lb = Cx - 0.1
    ub = Cx + 0.1
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    A_soc = jrnd.uniform(ka_soc, shape=(1, 2, dim), minval=-1, maxval=1)
    a_soc = jnp.zeros((1, 2, 1))
    f_soc = jrnd.uniform(kf, shape=(1, 1, dim), minval=-0.5, maxval=0.5)
    b_soc = jnp.ones((1, 1, 1))
    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        A=A_soc,
        a=a_soc,
        f=f_soc,
        b=b_soc,
    )
    nl_constraint = NonLinearConstraint(spec=nl_spec)

    parser = ConstraintParser(
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        box_constraint=None,
        nl_constraints=[nl_constraint],
    )
    eq_lifted, cart_lifted, lift_fn = parser.parse()

    assert eq_lifted is not None
    assert cart_lifted is not None
    assert cart_lifted.box_constraint is not None

    yraw = ProjectionInstance(x=x_ref, nl=[nl_spec])
    lifted = lift_fn(yraw)
    assert lifted.x.shape[1] > dim


def test_parse_non_linear_l2_norm_type_raises_not_implemented():
    """Test parser raises for L2 norm nonlinear constraints."""
    A = jnp.array([[[1.0, 0.0]]])
    b = jnp.array([[[0.0]]])
    C = jnp.array([[[1.0, 0.0]]])
    lb = jnp.array([[[-1.0]]])
    ub = jnp.array([[[1.0]]])

    eq_constraint = EqualityConstraint(A=A, b=b, var_b=False)
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    nl_spec = NonLinearSpecification(
        nl_type=L2NormType,
        A=jnp.array([[[1.0, 0.0], [0.0, 1.0]]]),
        a=jnp.zeros((1, 2, 1)),
        f=None,
        b=jnp.ones((1, 1, 1)),
    )
    nl_constraint = NonLinearConstraint(spec=nl_spec)

    parser = ConstraintParser(
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        box_constraint=None,
        nl_constraints=[nl_constraint],
    )

    with pytest.raises(NotImplementedError, match="L2NormType is not implemented"):
        parser.parse()


def test_parse_non_linear_irrelevant_type_raises_value_error():
    """Test parser raises for unsupported nonlinear type value."""
    A = jnp.array([[[1.0, 0.0]]])
    b = jnp.array([[[0.0]]])
    C = jnp.array([[[1.0, 0.0]]])
    lb = jnp.array([[[-1.0]]])
    ub = jnp.array([[[1.0]]])

    eq_constraint = EqualityConstraint(A=A, b=b, var_b=False)
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        A=jnp.array([[[1.0, 0.0]]]),
        a=jnp.zeros((1, 1, 1)),
        f=jnp.array([[[0.0, 1.0]]]),
        b=jnp.ones((1, 1, 1)),
    )
    nl_constraint = NonLinearConstraint(spec=nl_spec)
    nl_constraint._nl_type = "irrelevant_type"

    parser = ConstraintParser(
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        box_constraint=None,
        nl_constraints=[nl_constraint],
    )

    with pytest.raises(ValueError, match="Unsupported non-linear constraint type"):
        parser.parse()


def test_parse_non_linear_with_ineq_batch_size_not_one_raises():
    """Test parser raises when inequality C batch size is not 1 in nonlinear mode."""
    eq_constraint = EqualityConstraint(
        A=jnp.array([[[1.0, 0.0, 0.0]]]),
        b=jnp.array([[[0.0]]]),
        var_b=False,
    )

    # Nonlinear parsing requires C batch size == 1; use batch size 2 to trigger.
    C = jnp.array(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
        ]
    )
    lb = jnp.zeros((2, 1, 1))
    ub = jnp.ones((2, 1, 1))
    ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)

    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        A=jnp.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        a=jnp.zeros((1, 2, 1)),
        f=jnp.array([[[0.0, 0.0, 1.0]]]),
        b=jnp.ones((1, 1, 1)),
    )
    nl_constraint = NonLinearConstraint(spec=nl_spec)

    with pytest.raises(
        AssertionError,
        match="Batch size of inequality constraint C must be 1 or None",
    ):
        ConstraintParser(
            eq_constraint=eq_constraint,
            ineq_constraint=ineq_constraint,
            box_constraint=None,
            nl_constraints=[nl_constraint],
        )


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZES))
def test_only_nonlinear_constraints(seed, batch_size):
    """Test parser/project flow with only nonlinear constraints active."""
    dim = 3
    key = jrnd.PRNGKey(seed)

    key, kA, ka, kf, kx = jrnd.split(key, 5)
    A_soc = jrnd.uniform(kA, shape=(1, 2, dim), minval=-2, maxval=2)
    a_soc = jrnd.uniform(ka, shape=(1, 2, 1), minval=-1, maxval=1)
    f_soc = jrnd.uniform(kf, shape=(1, 1, dim), minval=-1, maxval=1)
    b_soc = jnp.ones((1, 1, 1))

    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        A=A_soc,
        a=a_soc,
        f=f_soc,
        b=b_soc,
    )
    nl_constraint = NonLinearConstraint(spec=nl_spec)

    parser = ConstraintParser(
        eq_constraint=None,
        ineq_constraint=None,
        box_constraint=None,
        nl_constraints=[nl_constraint],
    )
    eq_lifted, cart_lifted, lift_fn = parser.parse(method="pinv")

    # Build a random batch and run the ADMM projection loop.
    x = jrnd.uniform(kx, shape=(batch_size, dim, 1), minval=-3, maxval=3)
    yraw = ProjectionInstance(x=x, nl=[nl_spec])
    y_lifted = lift_fn(yraw)

    iteration_step, final_step = build_iteration_step(
        eq_constraint=eq_lifted,
        box_constraint=cart_lifted,
        dim=dim,
    )
    iteration_step = jax.jit(iteration_step)

    sk = ProjectionInstance(
        x=jnp.zeros((batch_size, y_lifted.x.shape[1], 1)),
        nl=[nl_spec],
    )
    for _ in range(200):
        sk = iteration_step(sk=sk, yraw=yraw, sigma=0.5, omega=1.7)

    yk = final_step(sk)

    # Nonlinear CV should decrease and be near feasible after projection.
    cv_before = cart_lifted.cv(y_lifted)
    cv_after = cart_lifted.cv(yk)
    assert jnp.mean(cv_after) <= jnp.mean(cv_before)
    assert jnp.all(cv_after < 1e-4)


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZES))
def test_box_and_nonlinear_constraints(seed, batch_size):
    """Test parser/project with box + nonlinear constraints (no ineq).

    This reaches code path where ineq_constraint is None (lines 317-319).
    """
    dim = 5
    key = jrnd.PRNGKey(seed)

    key, kbox_lb, kbox_ub, kA, ka, kf, kx = jrnd.split(key, 7)

    # Create box constraint
    box_lb = jrnd.uniform(kbox_lb, shape=(1, dim, 1), minval=-2, maxval=-0.5)
    box_ub = jrnd.uniform(kbox_ub, shape=(1, dim, 1), minval=0.5, maxval=2)
    box_spec = BoxConstraintSpecification(
        lb=box_lb,
        ub=box_ub,
        mask=jnp.ones(dim, dtype=jnp.bool_),
    )
    box_constraint = BoxConstraint(box_spec)

    # Create nonlinear constraint
    A_soc = jrnd.uniform(kA, shape=(1, 2, dim), minval=-2, maxval=2)
    a_soc = jrnd.uniform(ka, shape=(1, 2, 1), minval=-1, maxval=1)
    f_soc = jrnd.uniform(kf, shape=(1, 1, dim), minval=-1, maxval=1)
    b_soc = jnp.ones((1, 1, 1))

    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        A=A_soc,
        a=a_soc,
        f=f_soc,
        b=b_soc,
    )
    nl_constraint = NonLinearConstraint(spec=nl_spec)

    parser = ConstraintParser(
        eq_constraint=None,
        ineq_constraint=None,
        box_constraint=box_constraint,
        nl_constraints=[nl_constraint],
    )
    eq_lifted, cart_lifted, lift_fn = parser.parse(method="pinv")

    # Build a random batch
    x = jrnd.uniform(kx, shape=(batch_size, dim, 1), minval=-1.5, maxval=1.5)
    yraw = ProjectionInstance(x=x, nl=[nl_spec])
    y_lifted = lift_fn(yraw)

    iteration_step, final_step = build_iteration_step(
        eq_constraint=eq_lifted,
        box_constraint=cart_lifted,
        dim=dim,
    )
    iteration_step = jax.jit(iteration_step)

    sk = ProjectionInstance(
        x=jnp.zeros((batch_size, y_lifted.x.shape[1], 1)),
        nl=[nl_spec],
    )
    for _ in range(200):
        sk = iteration_step(sk=sk, yraw=yraw, sigma=0.5, omega=1.7)

    yk = final_step(sk)

    # Verify projection feasibility
    cv_after = cart_lifted.cv(yk)
    assert jnp.all(cv_after < 1e-4)
