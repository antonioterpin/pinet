"""Test vjp of the projection layer."""

from functools import partial
from itertools import product
from typing import cast

import cvxpy as cp
import jax
import jax.test_util
import matplotlib.pyplot as plt
import numpy as np
import pytest
from cvxpy.constraints.constraint import Constraint as CvxpyConstraint
from jax import numpy as jnp
from matplotlib.patches import Polygon

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    BoxConstraintSpecification,
    EqualityConstraint,
    Project,
    ProjectionInstance,
)

jax.config.update("jax_enable_x64", True)


def test_triangle():
    plot_results = False
    # bottom left corner
    xs_bl = [
        jnp.array([-0.5, -0.5]).reshape(1, 2),
        jnp.array([-0.5, -0.25]).reshape(1, 2),
        jnp.array([-0.25, -0.5]).reshape(1, 2),
        jnp.array([-0.25, -0.75]).reshape(1, 2),
        jnp.array([-0.75, -0.5]).reshape(1, 2),
    ]
    j_bl = jnp.zeros((1, 2, 2))
    # top right corner
    xs_tr = [
        jnp.array([1.5, 1.5]).reshape(1, 2),
        jnp.array([1.5, 1.25]).reshape(1, 2),
        jnp.array([1.25, 1.5]).reshape(1, 2),
        jnp.array([1.25, 1.75]).reshape(1, 2),
        jnp.array([1.75, 1.5]).reshape(1, 2),
    ]
    j_tr = jnp.zeros((1, 2, 2))
    # points to the center right
    xs_cr = [
        jnp.array([1.5, 0.5]).reshape(1, 2),
        jnp.array([1.5, 0.25]).reshape(1, 2),
        jnp.array([1.25, 0.5]).reshape(1, 2),
        jnp.array([1.25, 0.75]).reshape(1, 2),
        jnp.array([1.75, 0.5]).reshape(1, 2),
    ]
    j_cr = jnp.array([[0, 0], [0, 1]]).reshape(1, 2, 2)
    # points to the bottom right
    xs_br = [
        jnp.array([1.5, -0.5]).reshape(1, 2),
        jnp.array([1.5, -0.25]).reshape(1, 2),
        jnp.array([1.25, -0.5]).reshape(1, 2),
        jnp.array([1.25, -0.75]).reshape(1, 2),
        jnp.array([1.75, -0.5]).reshape(1, 2),
    ]
    j_br = jnp.zeros((1, 2, 2))
    # points to the bottom
    xs_b = [
        jnp.array([0.5, -0.5]).reshape(1, 2),
        jnp.array([0.5, -0.25]).reshape(1, 2),
        jnp.array([0.25, -0.5]).reshape(1, 2),
        jnp.array([0.25, -0.75]).reshape(1, 2),
        jnp.array([0.75, -0.5]).reshape(1, 2),
    ]
    j_b = jnp.array([[1, 0], [0, 0]]).reshape(1, 2, 2)
    # point to the top
    xs_t = [
        jnp.array([0.5, 1.25]).reshape(1, 2),
        jnp.array([0.25, 1.5]).reshape(1, 2),
        jnp.array([-0.5, 1.5]).reshape(1, 2),
        jnp.array([-0.5, 1.25]).reshape(1, 2),
        jnp.array([-0.25, 1.5]).reshape(1, 2),
        jnp.array([-0.25, 1.75]).reshape(1, 2),
        jnp.array([-0.75, 1.5]).reshape(1, 2),
        jnp.array([0.5, 0.75]).reshape(1, 2),
        jnp.array([0.25, 0.5]).reshape(1, 2),
        jnp.array([0.25, 0.75]).reshape(1, 2),
        jnp.array([0.75, 1]).reshape(1, 2),
    ]
    # This is something like 1/cos^2, and 1/cos*sin
    j_t = jnp.ones((1, 2, 2)) / 2.0
    # point in the triangle
    xs_in = [
        jnp.array([0.5, 0.25]).reshape(1, 2),
        jnp.array([0.75, 0.5]).reshape(1, 2),
    ]
    j_in = jnp.eye(2).reshape(1, 2, 2)

    if plot_results:
        # Plot the triangle
        _fig, ax = plt.subplots()
        triangle = Polygon(
            ((0, 0), (1, 0), (1, 1)), facecolor="blue", edgecolor="blue", alpha=0.5
        )
        ax.add_patch(triangle)
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
        ax.set_aspect("equal", adjustable="box")

        colors = ["black", "red", "green", "orange", "purple", "cyan", "magenta"]
        point_sets = [xs_bl, xs_tr, xs_cr, xs_br, xs_b, xs_t, xs_in]
        for color, points in zip(colors, point_sets, strict=True):
            for x in points:
                ax.plot(x[0, 0], x[0, 1], "o", color=color)

        plt.grid(True)
        plt.show()

    sigma = 1.0
    omega = 1.0
    # % Solve this with our projection layer
    box_constraint = BoxConstraint(
        BoxConstraintSpecification(
            lb=jnp.array([-jnp.inf, 0]).reshape(1, 2, 1),
            ub=jnp.array([1, jnp.inf]).reshape(1, 2, 1),
        )
    )
    affine_constraint = AffineInequalityConstraint(
        constr_matrix=jnp.array([-1, 1]).reshape(1, 1, 2),
        lb=jnp.array([-jnp.inf]).reshape(1, 1, 1),
        ub=jnp.zeros((1, 1, 1)),
    )

    projection_layer = Project(
        box_constraint=box_constraint,
        ineq_constraint=affine_constraint,
        unroll=False,
    )

    def fun_layer(x, v, fpi):
        return (
            projection_layer.call(
                yraw=ProjectionInstance(x=x[..., None]),
                sigma=sigma,
                omega=omega,
                n_iter=100,
                n_iter_bwd=100,
                fpi=fpi,
            )[0].x[..., 0]
            @ v
        ).mean()

    fun_layer = jax.jit(fun_layer, static_argnames=["fpi"])
    e_1 = jnp.eye(2)[0].reshape(2, 1)
    e_2 = jnp.eye(2)[1].reshape(2, 1)

    def j_x(x, fpi):
        grad_e1 = (jax.grad(fun_layer, argnums=0)(x, v=e_1, fpi=fpi)).reshape(1, 1, 2)
        grad_e2 = (jax.grad(fun_layer, argnums=0)(x, v=e_2, fpi=fpi)).reshape(1, 1, 2)
        return jnp.concatenate((grad_e1, grad_e2), axis=1)

    # Check the Jacobian of the projection
    for xs, j_true in zip(
        [xs_bl, xs_tr, xs_cr, xs_br, xs_b, xs_t, xs_in],
        [j_bl, j_tr, j_cr, j_br, j_b, j_t, j_in],
        strict=True,
    ):
        for x in xs:
            for fpi in [True, False]:
                jacobian = j_x(x, fpi)
                assert jnp.allclose(jacobian, j_true, atol=1e-4, rtol=1e-4), (
                    f"J={jacobian}, J_true={j_true} for x={x}"
                )


def test_box():
    plot_results = False
    # bottom left
    xs_bl = [
        jnp.array([-0.5, -0.5]).reshape(1, 2),
        jnp.array([-0.5, -0.25]).reshape(1, 2),
        jnp.array([-0.25, -0.75]).reshape(1, 2),
        jnp.array([-0.75, -0.5]).reshape(1, 2),
    ]
    j_bl = jnp.array([[1, 0], [0, 0]]).reshape(1, 2, 2)
    # top right corner
    xs_tr = [
        jnp.array([0.5, 0.5]).reshape(1, 2),
        jnp.array([0.5, 0.25]).reshape(1, 2),
        jnp.array([0.25, 0.5]).reshape(1, 2),
        jnp.array([0.25, 0.75]).reshape(1, 2),
        jnp.array([0.75, 0.5]).reshape(1, 2),
    ]
    j_tr = jnp.array([[0, 0], [0, 1]]).reshape(1, 2, 2)
    # points to the bottom right
    xs_br = [
        jnp.array([0.5, -0.5]).reshape(1, 2),
        jnp.array([0.5, -0.25]).reshape(1, 2),
        jnp.array([0.25, -0.5]).reshape(1, 2),
        jnp.array([0.25, -0.75]).reshape(1, 2),
        jnp.array([0.75, -0.5]).reshape(1, 2),
    ]
    j_br = jnp.zeros((1, 2, 2))
    # points to top left
    xs_tl = [
        jnp.array([-0.5, 0.5]).reshape(1, 2),
        jnp.array([-0.5, 0.25]).reshape(1, 2),
        jnp.array([-0.25, 0.5]).reshape(1, 2),
        jnp.array([-0.25, 0.75]).reshape(1, 2),
        jnp.array([-0.75, 0.5]).reshape(1, 2),
    ]
    j_tl = jnp.eye(2).reshape(1, 2, 2)

    if plot_results:
        _fig, ax = plt.subplots()
        quadrant = Polygon(
            ((-2, 0), (0, 0), (0, 2), (-2, 2)),
            facecolor="blue",
            edgecolor="blue",
            alpha=0.5,
        )
        ax.add_patch(quadrant)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal", adjustable="box")

        colors = ["black", "red", "green", "orange"]
        point_sets = [xs_bl, xs_tr, xs_br, xs_tl]
        for color, points in zip(colors, point_sets, strict=True):
            for x in points:
                ax.plot(x[0, 0], x[0, 1], "o", color=color)

        plt.grid(True)
        plt.show()

    lb = jnp.array([-jnp.inf, 0.0]).reshape((1, 2, 1))
    ub = jnp.array([0.0, jnp.inf]).reshape((1, 2, 1))
    box_constraint = BoxConstraint(BoxConstraintSpecification(lb=lb, ub=ub))
    projection_layer = Project(box_constraint=box_constraint)

    def fun_layer(x, v):
        return (
            projection_layer.call(yraw=ProjectionInstance(x=x[..., None]))[0].x[..., 0]
            @ v
        ).mean()

    e_1 = jnp.eye(2)[0]
    e_2 = jnp.eye(2)[1]

    def j_x(x):
        grad_e1 = (jax.grad(fun_layer, argnums=0)(x, e_1)).reshape(1, 1, 2)
        grad_e2 = (jax.grad(fun_layer, argnums=0)(x, e_2)).reshape(1, 1, 2)
        return jnp.concatenate((grad_e1, grad_e2), axis=1)

    # Check the Jacobian of the projection
    for xs, j_true in zip(
        [xs_bl, xs_tr, xs_br, xs_tl],
        [j_bl, j_tr, j_br, j_tl],
        strict=True,
    ):
        for x in xs:
            jacobian = j_x(x)
            assert jnp.allclose(jacobian, j_true, atol=1e-4, rtol=1e-4), (
                f"J={jacobian}, J_true={j_true} for x={x}"
            )


SEEDS = [8, 24, 42]
BATCH_SIZE = [1, 10]


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZE))
def test_general_eq_ineq(seed, batch_size):
    method = "pinv"
    dim = 100
    n_eq = 50
    n_ineq = 40
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=5)
    # Generate equality constraints LHS
    a_dyn = jax.random.normal(key[0], shape=(1, n_eq, dim))
    # Generate inequality constraints LHS
    constr_matrix = jax.random.normal(key[1], shape=(1, n_ineq, dim))
    # Compute RHS by solving feasibility problem
    xfeas = cp.Variable(dim)
    bfeas = cp.Variable(n_eq)
    lfeas = cp.Variable(n_ineq)
    ufeas = cp.Variable(n_ineq)
    a_dyn_cvx = cp.Constant(np.asarray(a_dyn[0, :, :]))
    constr_matrix_cvx = cp.Constant(np.asarray(constr_matrix[0, :, :]))
    feasibility_constraints: list[CvxpyConstraint] = [
        a_dyn_cvx @ xfeas == bfeas,
        lfeas <= constr_matrix_cvx @ xfeas,
        constr_matrix_cvx @ xfeas <= ufeas,
        -1 <= xfeas,
        xfeas <= 1,
    ]
    objective = cp.Minimize(jnp.ones(shape=(dim)) @ xfeas)
    problem = cp.Problem(objective=objective, constraints=feasibility_constraints)
    problem.solve(solver=cp.CLARABEL)
    # Extract RHS parameters
    b = jnp.tile(jnp.array(bfeas.value).reshape((1, n_eq, 1)), (1, 1, 1))
    lb = jnp.tile(jnp.array(lfeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))
    ub = jnp.tile(jnp.array(ufeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))

    # Define projection layer ingredients
    eq_constraint = EqualityConstraint(a_dyn=a_dyn, b=b, method=method)
    ineq_constraint = AffineInequalityConstraint(
        constr_matrix=constr_matrix, lb=lb, ub=ub
    )

    # Projection layer with unrolling differentiation
    projection_layer_unroll = Project(
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        unroll=True,
    )

    # Projection layer with implicit differentiation
    projection_layer_impl = Project(
        eq_constraint=eq_constraint,
        ineq_constraint=ineq_constraint,
        unroll=False,
    )

    # Point to be projected
    x = jax.random.uniform(key[2], shape=(batch_size, dim), minval=-2, maxval=2)

    # Compute the projection by solving QP
    yqp = jnp.zeros(shape=(batch_size, dim))
    for ii in range(batch_size):
        yproj = cp.Variable(dim)
        qp_constraints = cast(
            list[CvxpyConstraint],
            [
                a_dyn_cvx @ yproj == np.asarray(b[0, :, 0]),
                np.asarray(lb[0, :, 0]) <= constr_matrix_cvx @ yproj,
                constr_matrix_cvx @ yproj <= np.asarray(ub[0, :, 0]),
            ],
        )
        objective = cp.Minimize(cp.sum_squares(yproj - x[ii, :]))
        problem_qp = cp.Problem(objective=objective, constraints=qp_constraints)
        problem_qp.solve(
            solver=cp.CLARABEL, verbose=False, tol_gap_abs=1e-10, tol_gap_rel=1e-10
        )
        yqp = yqp.at[ii, :].set(jnp.array(yproj.value).reshape(dim))

    # Check that the projection are computed correctly
    n_iter = 200
    y_unroll = projection_layer_unroll.call(
        yraw=ProjectionInstance(x=x[..., None]),
        n_iter=n_iter,
    )[0].x[..., 0]
    y_impl = projection_layer_impl.call(
        yraw=ProjectionInstance(x=x[..., None]),
        n_iter=n_iter,
    )[0].x[..., 0]
    assert jnp.allclose(y_unroll, yqp, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(y_impl, yqp, atol=1e-4, rtol=1e-4)

    # Simple "loss" function as inner product
    n_iter = 500
    vec = jnp.array(jax.random.normal(key[3], shape=(dim, batch_size)))

    def loss(x, v, unroll, n_iter_bwd, fpi):
        inp = ProjectionInstance(x=x[..., None])
        if unroll:
            return (
                projection_layer_unroll.call(yraw=inp, n_iter=n_iter)[0].x[..., 0] @ v
            ).mean()
        else:
            return (
                projection_layer_impl.call(
                    yraw=inp,
                    n_iter=n_iter,
                    n_iter_bwd=n_iter_bwd,
                    fpi=fpi,
                )[0].x[..., 0]
                @ v
            ).mean()

    grad_unroll = jax.vmap(
        lambda x, vec: jax.grad(loss, argnums=0)(
            x.reshape(1, dim), vec.reshape(dim, 1), True, n_iter_bwd=-1, fpi=True
        ),
        in_axes=[0, 1],
    )(x, vec).reshape(batch_size, -1)
    grad_fpi = jax.vmap(
        lambda x, vec: jax.grad(loss, argnums=0)(
            x.reshape(1, dim), vec.reshape(dim, 1), False, n_iter_bwd=200, fpi=True
        ),
        in_axes=[0, 1],
    )(x, vec).reshape(batch_size, -1)
    grad_linsys = jax.vmap(
        lambda x, vec: jax.grad(loss, argnums=0)(
            x.reshape(1, dim), vec.reshape(dim, 1), False, n_iter_bwd=50, fpi=False
        ),
        in_axes=[0, 1],
    )(x, vec).reshape(batch_size, -1)
    assert jnp.allclose(grad_unroll, grad_fpi, atol=1e-4, rtol=1e-4)
    assert jnp.allclose(grad_unroll, grad_linsys, atol=1e-4, rtol=1e-4)

    # Compute the gradient with finite differences
    epsilon = 1e-5
    # Random direction
    dir = jax.random.uniform(key[4], shape=(batch_size, dim), minval=-1, maxval=1)
    direps = dir * epsilon
    # Compute the loss
    lossvmapped = jax.vmap(
        lambda x, vec: loss(
            x.reshape(1, dim), vec.reshape(dim, 1), True, n_iter_bwd=-1, fpi=True
        ),
        in_axes=[0, 1],
    )
    thelossp = lossvmapped(x + direps, vec)
    thelossm = lossvmapped(x - direps, vec)
    grad_fd = (thelossp - thelossm) / (2 * epsilon)
    for name, grad in zip(
        ["grad_unroll", "grad_fpi", "grad_linsys"],
        [grad_unroll, grad_fpi, grad_linsys],
        strict=True,
    ):
        dirgrad = jnp.vecdot(grad, dir)
        if not jnp.allclose(dirgrad, grad_fd, atol=1e-3, rtol=1e-3):
            print(
                f"Assertion failed for {name}: dirgrad = {dirgrad}, grad_fd = {grad_fd}"
            )
            raise AssertionError(f"Assertion failed for {name}")

    # Use jax utils for checking
    def unroll_f(y):
        return projection_layer_unroll.call(
            yraw=ProjectionInstance(x=y),
            n_iter=200,
        )[0].x[..., 0]

    def fpi_f(y):
        return projection_layer_impl.call(
            yraw=ProjectionInstance(x=y),
            n_iter=200,
            n_iter_bwd=200,
            fpi=True,
        )[0].x[..., 0]

    def linsys_f(y):
        return projection_layer_impl.call(
            yraw=ProjectionInstance(x=y),
            n_iter=200,
            n_iter_bwd=50,
            fpi=False,
        )[0].x[..., 0]

    for name, f in [("unroll_f", unroll_f), ("fpi_f", fpi_f), ("linsys_f", linsys_f)]:
        try:
            jax.test_util.check_vjp(
                f,
                partial(jax.vjp, f),
                (x[..., None],),
                eps=1e-5,
                atol=1e-3,
                rtol=1e-3,
            )
        except AssertionError as err:
            print(f"Assertion failed for {name}: {err}")
