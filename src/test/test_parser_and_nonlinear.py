"""Tests for the constraint parser with non-linear constraints.

Covers parsing and projection when non-linear constraints (L2-norm / SOC
types) are combined with equality, inequality, and box constraints,
including the ``var_a_mat=True`` per-instance path.
"""

from itertools import product
from typing import cast

import cvxpy as cp
import jax
import jax.numpy as jnp
import jax.random as jrnd
import numpy as np
import pytest
from cvxpy.constraints.constraint import Constraint as CvxpyConstraint

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    BoxConstraintSpecification,
    CartesianConstraint,
    ConstraintParser,
    EqualityConstraint,
    EqualityConstraintsSpecification,
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
    n_a = 1
    n_c = 2
    n_a_soc_1 = 3
    n_a_soc_2 = 4
    key = jrnd.PRNGKey(seed)
    # Generate a random point which will be feasible by-construction
    key, subkey = jrnd.split(key)
    x_feas = jrnd.uniform(subkey, shape=(1, dim, 1), minval=-2, maxval=2)

    # Equality constraint
    a_mat = jrnd.uniform(key, shape=(1, n_a, dim), minval=-2, maxval=2)
    b = a_mat @ x_feas
    eq_constraint = EqualityConstraint(a_mat=a_mat, b=b, var_b=False)

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
    a_soc_1 = jrnd.uniform(subkey, shape=(1, n_a_soc_1, 1), minval=0.5, maxval=2)
    key, subkey = jrnd.split(key)
    f_soc_1 = jrnd.uniform(subkey, shape=(1, 1, dim), minval=0, maxval=1)
    b_soc_1 = (
        eps_soc
        + jnp.linalg.norm(a_soc_1_mat @ x_feas + a_soc_1, ord=2, axis=1)
        - f_soc_1 @ x_feas
    )
    nlspec_1 = NonLinearSpecification(
        a_mat=a_soc_1_mat,
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
    a_soc_2_mat = jrnd.uniform(subkey, shape=(1, n_a_soc_2, dim), minval=-2, maxval=2)
    key, subkey = jrnd.split(key)
    a_soc_2 = jrnd.uniform(subkey, shape=(1, n_a_soc_2, 1), minval=0.5, maxval=2)
    key, subkey = jrnd.split(key)
    f_soc_2 = jrnd.uniform(subkey, shape=(1, 1, dim), minval=-1, maxval=1)
    b_soc_2 = (
        eps_soc
        + jnp.linalg.norm(a_soc_2_mat @ x_feas + a_soc_2, ord=2, axis=1)
        - f_soc_2 @ x_feas
    )
    nlspec_2 = NonLinearSpecification(
        a_mat=a_soc_2_mat,
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
    # The non-linear path returns a lifted equality and a cartesian constraint.
    assert eq_lifted is not None
    assert isinstance(cart_lifted, CartesianConstraint)

    # Check parsing
    # Dimension of augmented space
    # n_aug = dim + n_C + n_A_soc_1 + 1 + n_A_soc_2 + 1
    # Extra dimensions in augmented
    n_extra = n_c + n_a_soc_1 + 1 + n_a_soc_2 + 1
    a_lifted_correct_mat = jnp.concatenate(
        [
            jnp.concatenate(
                [a_mat, c_mat, a_soc_1_mat, f_soc_1, a_soc_2_mat, f_soc_2], axis=1
            ),
            jnp.zeros((1, n_a + n_extra, n_extra)),
        ],
        axis=2,
    )
    a_lifted_correct_mat = a_lifted_correct_mat + jnp.zeros_like(a_lifted_correct_mat).at[
        :, n_a:, dim:
    ].set(-jnp.eye(n_extra))
    b_lifted_correct = jnp.concatenate([b, jnp.zeros((1, n_extra, 1))], axis=1)
    mask_box_correct = jnp.concatenate(
        [mask, jnp.array([True] * n_c + [False] * (n_extra - n_c), dtype=jnp.bool_)]
    )
    box_ub_correct = jnp.concatenate([ub_box, ub_ineq], axis=1)
    box_lb_correct = jnp.concatenate([lb_box, lb_ineq], axis=1)
    soc_1_mask_u_correct = jnp.array(
        [False] * (dim + n_c) + [True] * n_a_soc_1 + [False] * (1 + n_a_soc_2 + 1),
        dtype=jnp.bool_,
    )
    soc_1_mask_t_correct = jnp.array(
        [False] * (dim + n_c + n_a_soc_1) + [True] + [False] * (n_a_soc_2 + 1),
        dtype=jnp.bool_,
    )
    soc_2_mask_u_correct = jnp.array(
        [False] * (dim + n_c + n_a_soc_1 + 1) + [True] * n_a_soc_2 + [False],
        dtype=jnp.bool_,
    )
    soc_2_mask_t_correct = jnp.array(
        [False] * (dim + n_c + n_a_soc_1 + 1 + n_a_soc_2) + [True], dtype=jnp.bool_
    )

    # Assertions
    assert jnp.allclose(eq_lifted.a_mat, a_lifted_correct_mat), (
        """Lifted A matrix is incorrect."""
    )
    assert jnp.allclose(eq_lifted.b, b_lifted_correct), """
        Lifted b vector is incorrect.
    """
    assert isinstance(cart_lifted.constraints[0], BoxConstraint), """
        First constraint should be BoxConstraint.
    """
    box_first = cart_lifted.constraints[0]
    assert isinstance(box_first, BoxConstraint)
    assert box_first.lb is not None and box_first.ub is not None
    assert jnp.allclose(box_first.mask, mask_box_correct), """
        Box mask is incorrect.
    """
    assert jnp.allclose(box_first.ub, box_ub_correct), """
        Box upper bound is incorrect.
    """
    assert jnp.allclose(box_first.lb, box_lb_correct), """
        Box lower bound is incorrect.
    """
    assert isinstance(cart_lifted.constraints[1], SocConstraint), """
        Second constraint should be SocConstraint.
    """
    assert jnp.allclose(cart_lifted.constraints[1].mask_u, soc_1_mask_u_correct), """
        SOC 1 mask_u is incorrect.
    """
    assert jnp.allclose(cart_lifted.constraints[1].mask_t, soc_1_mask_t_correct), """
        SOC 1 mask_t is incorrect.
    """
    assert isinstance(cart_lifted.constraints[2], SocConstraint), """
        Third constraint should be SocConstraint.
    """
    assert jnp.allclose(cart_lifted.constraints[2].mask_u, soc_2_mask_u_correct), """
        SOC 2 mask_u is incorrect.
    """
    assert jnp.allclose(cart_lifted.constraints[2].mask_t, soc_2_mask_t_correct), """
        SOC 2 mask_t is incorrect.
    """

    # Create random points to be projected
    key, subkey = jrnd.split(key)
    yproj = jrnd.uniform(subkey, shape=(batch_size, dim, 1), minval=-5, maxval=5)
    y_raw = ProjectionInstance(x=yproj, nl=[nlspec_1, nlspec_2])

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
    for _ii in range(n_iter):
        sk = iteration_step(sk=sk, y_raw=y_raw, sigma=0.1, omega=1.8)
    yk = final_step(sk)

    # Compute projection with cvxpy
    y_cvxpy = cp.Variable(dim)
    x_cvxpy = cp.Parameter(dim)
    constraints = [
        a_mat[0, :, :] @ y_cvxpy == b[0, :, 0],
        lb_box[0, :, 0] <= y_cvxpy[mask],
        y_cvxpy[mask] <= ub_box[0, :, 0],
        lb_ineq[0, :, 0] <= c_mat[0, :, :] @ y_cvxpy,
        c_mat[0, :, :] @ y_cvxpy <= ub_ineq[0, :, 0],
        cp.SOC(
            f_soc_1[0, :, :] @ y_cvxpy + b_soc_1[0, :, 0],
            a_soc_1_mat[0, :, :] @ y_cvxpy + a_soc_1[0, :, 0],
        ),
        cp.SOC(
            f_soc_2[0, :, :] @ y_cvxpy + b_soc_2[0, :, 0],
            a_soc_2_mat[0, :, :] @ y_cvxpy + a_soc_2[0, :, 0],
        ),
    ]
    objective = cp.Minimize(cp.sum_squares(y_cvxpy - x_cvxpy))
    problem_cvxpy = cp.Problem(
        objective=objective, constraints=cast(list[CvxpyConstraint], constraints)
    )
    y_opt = jnp.zeros((batch_size, dim, 1))
    for ii in range(batch_size):
        x_cvxpy.value = np.array(yproj[ii].reshape(-1))
        problem_cvxpy.solve(solver=cp.SCS, verbose=False, eps_abs=1e-9, eps_rel=1e-9)
        y_opt = y_opt.at[ii].set(jnp.array(y_cvxpy.value).reshape(-1, 1))

    assert jnp.allclose(yk.x[:, :dim, :], y_opt, atol=1e-5, rtol=1e-5), """
        Projected points do not match CVXPY solution.
    """
    assert jnp.allclose(
        yk.x[:, dim:, :], eq_lifted.a_mat[0, n_a:, :dim] @ y_opt, atol=1e-5, rtol=1e-5
    ), """
        Auxiliary variables do not match CVXPY solution.
    """

    # Resolve with different values for a_soc_2 and b_soc_2
    key, subkey = jrnd.split(key)
    a_soc_2_new = jrnd.uniform(subkey, shape=(1, n_a_soc_2, 1), minval=0.5, maxval=2)
    key, subkey = jrnd.split(key)
    b_soc_2_new = (
        eps_soc
        + jnp.linalg.norm(a_soc_2_mat @ x_feas + a_soc_2_new, ord=2, axis=1)
        - f_soc_2 @ x_feas
    )
    nlspec_2_new = nlspec_2.update(a=a_soc_2_new, b=b_soc_2_new)
    y_raw_new = ProjectionInstance(x=yproj, nl=[nlspec_1, nlspec_2_new])
    sk = ProjectionInstance(
        x=jnp.zeros((batch_size, dim + n_extra, 1)), nl=[nlspec_1, nlspec_2_new]
    )
    for _ii in range(n_iter):
        sk = iteration_step(sk=sk, y_raw=y_raw_new, sigma=0.1, omega=1.8)
    yk_new = final_step(sk)

    # CVXPY solution with updated SOC parameters
    constraints[6] = cp.SOC(
        f_soc_2[0, :, :] @ y_cvxpy + b_soc_2_new[0, :, 0],
        a_soc_2_mat[0, :, :] @ y_cvxpy + a_soc_2_new[0, :, 0],
    )
    problem_cvxpy = cp.Problem(
        objective=objective, constraints=cast(list[CvxpyConstraint], constraints)
    )
    y_opt_new = jnp.zeros((batch_size, dim, 1))
    for ii in range(batch_size):
        x_cvxpy.value = np.array(yproj[ii].reshape(-1))
        problem_cvxpy.solve(solver=cp.SCS, verbose=False, eps_abs=1e-9, eps_rel=1e-9)
        y_opt_new = y_opt_new.at[ii].set(jnp.array(y_cvxpy.value).reshape(-1, 1))

    assert jnp.allclose(yk_new.x[:, :dim, :], y_opt_new, atol=1e-5, rtol=1e-5), """
        Projected points do not match CVXPY solution.
    """
    assert jnp.allclose(
        yk_new.x[:, dim:, :],
        eq_lifted.a_mat[0, n_a:, :dim] @ y_opt_new,
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
    a_mat = jrnd.uniform(ka, shape=(1, n_eq, dim), minval=-1, maxval=1)
    x_ref = jrnd.uniform(ks, shape=(1, dim, 1), minval=-1, maxval=1)
    b = a_mat @ x_ref
    eq_constraint = EqualityConstraint(a_mat=a_mat, b=b, var_b=False)

    c_mat = jrnd.uniform(kc, shape=(1, n_ineq, dim), minval=-1, maxval=1)
    Cx = c_mat @ x_ref
    lb = Cx - 0.1
    ub = Cx + 0.1
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)

    a_soc_mat = jrnd.uniform(ka_soc, shape=(1, 2, dim), minval=-1, maxval=1)
    a_soc = jnp.zeros((1, 2, 1))
    f_soc = jrnd.uniform(kf, shape=(1, 1, dim), minval=-0.5, maxval=0.5)
    b_soc = jnp.ones((1, 1, 1))
    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=a_soc_mat,
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
    assert isinstance(cart_lifted, CartesianConstraint)
    assert cart_lifted.box_constraint is not None

    y_raw = ProjectionInstance(x=x_ref, nl=[nl_spec])
    lifted = lift_fn(y_raw)
    assert lifted.x.shape[1] > dim


def test_parse_non_linear_var_b_extends_rhs():
    """With ``var_b=True`` the non-linear lift extends the per-instance ``b``.

    The lift closure must append the auxiliary-variable rows to ``y.eq.b`` so
    the runtime RHS matches the static ``b_lifted`` built by the parser. The
    appended block is zeros (the auxiliary equalities ``u_aux = A x`` have no
    constant term), so the lifted ``b`` is the original RHS followed by zeros.
    """
    dim = 3
    n_eq = 1
    n_ineq = 2
    key = jrnd.PRNGKey(1)

    key, ka, kc, ks, kf, ka_soc = jrnd.split(key, 6)
    a_mat = jrnd.uniform(ka, shape=(1, n_eq, dim), minval=-1, maxval=1)
    x_ref = jrnd.uniform(ks, shape=(1, dim, 1), minval=-1, maxval=1)
    b = a_mat @ x_ref
    eq_constraint = EqualityConstraint(a_mat=a_mat, b=b, var_b=True)

    c_mat = jrnd.uniform(kc, shape=(1, n_ineq, dim), minval=-1, maxval=1)
    Cx = c_mat @ x_ref
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=Cx - 0.1, ub=Cx + 0.1)

    a_soc_mat = jrnd.uniform(ka_soc, shape=(1, 2, dim), minval=-1, maxval=1)
    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=a_soc_mat,
        a=jnp.zeros((1, 2, 1)),
        f=jrnd.uniform(kf, shape=(1, 1, dim), minval=-0.5, maxval=0.5),
        b=jnp.ones((1, 1, 1)),
    )
    nl_constraint = NonLinearConstraint(spec=nl_spec)

    parser = ConstraintParser(
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        box_constraint=None,
        nl_constraints=[nl_constraint],
    )
    eq_lifted, _, lift_fn = parser.parse()

    # The per-instance equality spec carries the variable RHS ``b``.
    y_raw = ProjectionInstance(
        x=x_ref,
        eq=EqualityConstraintsSpecification(b=b),
        nl=[nl_spec],
    )
    lifted = lift_fn(y_raw)

    # The lifted runtime RHS must match the static lifted RHS in shape: the
    # original ``n_eq`` rows plus one row per auxiliary variable.
    lifted_eq = lifted.eq
    assert lifted_eq is not None
    lifted_b = lifted_eq.b
    assert lifted_b is not None
    assert eq_lifted is not None
    static_b = eq_lifted.b
    assert static_b is not None
    assert lifted_b.shape == static_b.shape
    n_aux = static_b.shape[1] - n_eq
    assert n_aux > 0
    assert jnp.allclose(lifted_b[:, :n_eq, :], b)
    assert jnp.allclose(lifted_b[:, n_eq:, :], 0.0)


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZES))
def test_parse_non_linear_l2_norm_projection(seed: int, batch_size: int):
    """L2NormType lifts to SOC and the projected point satisfies the L2 ball."""
    dim = 4
    m = 3
    key = jrnd.PRNGKey(seed)

    key, ka_mat, ka, kx = jrnd.split(key, 4)
    a_l2_mat = jrnd.uniform(ka_mat, shape=(1, m, dim), minval=-1.0, maxval=1.0)
    a_l2 = jrnd.uniform(ka, shape=(1, m, 1), minval=-0.5, maxval=0.5)
    b_l2 = jnp.full((1, 1, 1), 1.5)  # constant scalar bound

    nl_spec = NonLinearSpecification(
        nl_type=L2NormType,
        a_mat=a_l2_mat,
        a=a_l2,
        f=None,
        b=b_l2,
    )
    nl_constraint = NonLinearConstraint(spec=nl_spec)

    parser = ConstraintParser(
        eq_constraint=None,
        ineq_constraint=None,
        box_constraint=None,
        nl_constraints=[nl_constraint],
    )
    eq_lifted, cart_lifted, lift_fn = parser.parse(method="pinv")
    # Narrow parser outputs before using them downstream.
    assert eq_lifted is not None
    assert isinstance(cart_lifted, CartesianConstraint)

    # Project a random batch and confirm the L2 ball constraint is satisfied.
    x = jrnd.uniform(kx, shape=(batch_size, dim, 1), minval=-3.0, maxval=3.0)
    y_raw = ProjectionInstance(x=x, nl=[nl_spec])
    y_lifted = lift_fn(y_raw)

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
    for _ in range(300):
        sk = iteration_step(sk=sk, y_raw=y_raw, sigma=0.5, omega=1.7)

    yk = final_step(sk)

    # ``||A x + a||_2 <= b`` should hold on the projected primal x.
    primal = yk.x[:, :dim, :]
    norms = jnp.linalg.norm(a_l2_mat @ primal + a_l2, axis=1, keepdims=True)
    assert jnp.all(norms <= b_l2 + 1e-3), (
        f"L2 norm bound violated after projection: max={float(jnp.max(norms))}, "
        f"bound={float(b_l2[0, 0, 0])}"
    )


def test_parse_non_linear_with_ineq_batch_size_not_one_raises():
    """Test parser raises when inequality C batch size is not 1 in nonlinear mode."""
    eq_constraint = EqualityConstraint(
        a_mat=jnp.array([[[1.0, 0.0, 0.0]]]),
        b=jnp.array([[[0.0]]]),
        var_b=False,
    )

    # Nonlinear parsing requires C batch size == 1; use batch size 2 to trigger.
    c_mat = jnp.array(
        [
            [[1.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0]],
        ]
    )
    lb = jnp.zeros((2, 1, 1))
    ub = jnp.ones((2, 1, 1))
    ineq_constraint = AffineInequalityConstraint(c_mat=c_mat, lb=lb, ub=ub)

    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=jnp.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        a=jnp.zeros((1, 2, 1)),
        f=jnp.array([[[0.0, 0.0, 1.0]]]),
        b=jnp.ones((1, 1, 1)),
    )
    nl_constraint = NonLinearConstraint(spec=nl_spec)

    with pytest.raises(
        AssertionError,
        match=r"Batch size of inequality constraint must be 1 or None",
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

    key, ka_mat, ka, kf, kx = jrnd.split(key, 5)
    a_soc_mat = jrnd.uniform(ka_mat, shape=(1, 2, dim), minval=-2, maxval=2)
    a_soc = jrnd.uniform(ka, shape=(1, 2, 1), minval=-1, maxval=1)
    f_soc = jrnd.uniform(kf, shape=(1, 1, dim), minval=-1, maxval=1)
    b_soc = jnp.ones((1, 1, 1))

    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=a_soc_mat,
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
    # Narrow parser outputs before using them downstream.
    assert eq_lifted is not None
    assert isinstance(cart_lifted, CartesianConstraint)

    # Build a random batch and run the ADMM projection loop.
    x = jrnd.uniform(kx, shape=(batch_size, dim, 1), minval=-3, maxval=3)
    y_raw = ProjectionInstance(x=x, nl=[nl_spec])
    y_lifted = lift_fn(y_raw)

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
        sk = iteration_step(sk=sk, y_raw=y_raw, sigma=0.5, omega=1.7)

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

    key, kbox_lb, kbox_ub, ka_mat, ka, kf, kx = jrnd.split(key, 7)

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
    a_soc_mat = jrnd.uniform(ka_mat, shape=(1, 2, dim), minval=-2, maxval=2)
    a_soc = jrnd.uniform(ka, shape=(1, 2, 1), minval=-1, maxval=1)
    f_soc = jrnd.uniform(kf, shape=(1, 1, dim), minval=-1, maxval=1)
    b_soc = jnp.ones((1, 1, 1))

    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=a_soc_mat,
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
    # Narrow parser outputs before using them downstream.
    assert eq_lifted is not None
    assert isinstance(cart_lifted, CartesianConstraint)

    # Build a random batch
    x = jrnd.uniform(kx, shape=(batch_size, dim, 1), minval=-1.5, maxval=1.5)
    y_raw = ProjectionInstance(x=x, nl=[nl_spec])
    y_lifted = lift_fn(y_raw)

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
        sk = iteration_step(sk=sk, y_raw=y_raw, sigma=0.5, omega=1.7)

    yk = final_step(sk)

    # Verify projection feasibility
    cv_after = cart_lifted.cv(yk)
    assert jnp.all(cv_after < 1e-4)


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, [1, 5]))
def test_project_var_a_mat_with_nonlinear(seed: int, batch_size: int):
    """``var_a_mat=True`` works on the non-linear path.

    The lifted ``a_mat`` propagates the user-supplied per-instance
    equality matrix through ``parse_non_linear`` (re-run inside
    ``solver.admm.initialize``), and the projection still satisfies
    both the equality and SOC constraints.
    """
    from pinet import (  # noqa: PLC0415  -- isolate the heavy import to this test
        EqualityConstraintsSpecification,
        Project,
    )

    dim = 4
    n_eq = 2
    m_soc = 3
    key = jrnd.PRNGKey(seed)

    key, ka_mat, kx_feas, kxinfeas, ka, kf = jrnd.split(key, 6)

    # Per-instance equality matrix and a feasible point that satisfies it.
    a_mat = jrnd.uniform(ka_mat, shape=(batch_size, n_eq, dim), minval=-1.0, maxval=1.0)
    x_feas = jrnd.uniform(kx_feas, shape=(batch_size, dim, 1), minval=-1.0, maxval=1.0)
    b_eq = a_mat @ x_feas

    # Build the SOC: ``||A_soc x + a_soc||_2 <= f_soc x + b_soc``.
    a_soc_mat = jrnd.uniform(ka_mat, shape=(1, m_soc, dim), minval=-1.0, maxval=1.0)
    a_soc = jrnd.uniform(ka, shape=(1, m_soc, 1), minval=-0.5, maxval=0.5)
    f_soc = jrnd.uniform(kf, shape=(1, 1, dim), minval=-0.5, maxval=0.5)
    soc_norm_at_feas = jnp.linalg.norm(a_soc_mat @ x_feas + a_soc, axis=1, keepdims=True)
    f_at_feas = f_soc @ x_feas
    # Pick ``b_soc`` per-instance so the feasible point is on the SOC's interior.
    b_soc = soc_norm_at_feas - f_at_feas + 0.5

    nl_spec = NonLinearSpecification(
        nl_type=SOCType,
        a_mat=a_soc_mat,
        a=a_soc,
        f=f_soc,
        b=b_soc,
    )
    nl_constraint = NonLinearConstraint(spec=nl_spec)

    # Construct ``Project`` with ``var_a_mat=True`` so the per-instance
    # ``a_mat`` is supplied at projection time via ``y_raw.eq``.
    eq_constraint = EqualityConstraint(
        a_mat=a_mat, b=b_eq, method="pinv", var_a_mat=True, var_b=True
    )
    project_layer = Project(
        eq_constraint=eq_constraint,
        nl_constraints=[nl_constraint],
    )

    # Project an infeasible point.
    x_infeas = jrnd.uniform(kxinfeas, shape=(batch_size, dim, 1), minval=-3.0, maxval=3.0)
    y_raw = ProjectionInstance(
        x=x_infeas,
        eq=EqualityConstraintsSpecification(a_mat=a_mat, b=b_eq),
        nl=[nl_spec],
    )
    yk, _ = project_layer.call(y_raw=y_raw, n_iter=500)
    primal = yk.x[:, :dim, :]

    # Equality holds.
    assert jnp.allclose(a_mat @ primal, b_eq, atol=1e-4), (
        "Per-instance equality constraint violated under var_a_mat=True."
    )
    # SOC holds.
    soc_lhs = jnp.linalg.norm(a_soc_mat @ primal + a_soc, axis=1, keepdims=True)
    soc_rhs = f_soc @ primal + b_soc
    assert jnp.all(soc_lhs <= soc_rhs + 1e-3), (
        f"SOC constraint violated under var_a_mat=True: "
        f"max LHS-RHS={float(jnp.max(soc_lhs - soc_rhs))}"
    )
