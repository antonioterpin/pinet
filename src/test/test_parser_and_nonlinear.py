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
