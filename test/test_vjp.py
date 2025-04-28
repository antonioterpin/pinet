"""Test vjp of the projection layer."""

import jax
from jax import numpy as jnp

from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.constraints.box import BoxConstraint
from hcnn.project import Project

jax.config.update("jax_enable_x64", True)


# TODO: Add tests for generic polyhedral set.
def test_triangle():
    PLOT_RESULTS = False
    # bottom left corner
    xs_bl = [
        jnp.array([-0.5, -0.5]).reshape(1, 2),
        jnp.array([-0.5, -0.25]).reshape(1, 2),
        jnp.array([-0.25, -0.5]).reshape(1, 2),
        jnp.array([-0.25, -0.75]).reshape(1, 2),
        jnp.array([-0.75, -0.5]).reshape(1, 2),
    ]
    J_bl = jnp.zeros((1, 2, 2))
    # top right corner
    xs_tr = [
        jnp.array([1.5, 1.5]).reshape(1, 2),
        jnp.array([1.5, 1.25]).reshape(1, 2),
        jnp.array([1.25, 1.5]).reshape(1, 2),
        jnp.array([1.25, 1.75]).reshape(1, 2),
        jnp.array([1.75, 1.5]).reshape(1, 2),
    ]
    J_tr = jnp.zeros((1, 2, 2))
    # points to the center right
    xs_cr = [
        jnp.array([1.5, 0.5]).reshape(1, 2),
        jnp.array([1.5, 0.25]).reshape(1, 2),
        jnp.array([1.25, 0.5]).reshape(1, 2),
        jnp.array([1.25, 0.75]).reshape(1, 2),
        jnp.array([1.75, 0.5]).reshape(1, 2),
    ]
    J_cr = jnp.array([[0, 0], [0, 1]]).reshape(1, 2, 2)
    # points to the bottom right
    xs_br = [
        jnp.array([1.5, -0.5]).reshape(1, 2),
        jnp.array([1.5, -0.25]).reshape(1, 2),
        jnp.array([1.25, -0.5]).reshape(1, 2),
        jnp.array([1.25, -0.75]).reshape(1, 2),
        jnp.array([1.75, -0.5]).reshape(1, 2),
    ]
    J_br = jnp.zeros((1, 2, 2))
    # points to the bottom
    xs_b = [
        jnp.array([0.5, -0.5]).reshape(1, 2),
        jnp.array([0.5, -0.25]).reshape(1, 2),
        jnp.array([0.25, -0.5]).reshape(1, 2),
        jnp.array([0.25, -0.75]).reshape(1, 2),
        jnp.array([0.75, -0.5]).reshape(1, 2),
    ]
    J_b = jnp.array([[1, 0], [0, 0]]).reshape(1, 2, 2)
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
    J_t = jnp.ones((1, 2, 2)) / 2.0
    # point in the triangle
    xs_in = [
        jnp.array([0.5, 0.25]).reshape(1, 2),
        jnp.array([0.75, 0.5]).reshape(1, 2),
    ]
    J_in = jnp.eye(2).reshape(1, 2, 2)

    if PLOT_RESULTS:
        # Plot the triangle
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        triangle = plt.Polygon(
            ((0, 0), (1, 0), (1, 1)), facecolor="blue", edgecolor="blue", alpha=0.5
        )
        ax.add_patch(triangle)
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
        ax.set_aspect("equal", adjustable="box")

        colors = ["black", "red", "green", "orange", "purple", "cyan", "magenta"]
        point_sets = [xs_bl, xs_tr, xs_cr, xs_br, xs_b, xs_t, xs_in]
        for color, points in zip(colors, point_sets):
            for x in points:
                ax.plot(x[0, 0], x[0, 1], "o", color=color)

        plt.grid(True)
        plt.show()

    sigma = 1.0
    omega = 1.0
    # % Solve this with our projection layer
    box_constraint = BoxConstraint(
        lower_bound=jnp.array([-jnp.inf, 0]).reshape(1, 2, 1),
        upper_bound=jnp.array([1, jnp.inf]).reshape(1, 2, 1),
    )
    affine_constraint = AffineInequalityConstraint(
        C=jnp.array([-1, 1]).reshape(1, 1, 2),
        lb=jnp.array([-jnp.inf]).reshape(1, 1, 1),
        ub=jnp.zeros((1, 1, 1)),
    )

    projection_layer = Project(
        box_constraint=box_constraint,
        ineq_constraint=affine_constraint,
        unroll=False,
        sigma=sigma,
        omega=omega,
    )
    fun_layer = jax.jit(
        lambda x, v, fpi: (
            projection_layer.call(x, 0.0, 100, n_iter_bwd=100, fpi=fpi)[0] @ v
        ).mean(),
        static_argnames=["fpi"],
    )
    e_1 = jnp.eye(2)[0].reshape(2, 1)
    e_2 = jnp.eye(2)[1].reshape(2, 1)

    def J_x(x, fpi):
        grad_e1 = (jax.grad(fun_layer, argnums=0)(x, e_1, fpi)).reshape(1, 1, 2)
        grad_e2 = (jax.grad(fun_layer, argnums=0)(x, e_2, fpi)).reshape(1, 1, 2)
        return jnp.concatenate((grad_e1, grad_e2), axis=1)

    # Check the Jacobian of the projection
    for xs, J_true in zip(
        [xs_bl, xs_tr, xs_cr, xs_br, xs_b, xs_t, xs_in],
        [J_bl, J_tr, J_cr, J_br, J_b, J_t, J_in],
    ):
        for x in xs:
            for fpi in [True, False]:
                J = J_x(x, fpi)
                assert jnp.allclose(
                    J, J_true, atol=1e-4, rtol=1e-4
                ), f"J={J}, J_true={J_true} for x={x}"


def test_box():
    PLOT_RESULTS = False
    # bottom left
    xs_bl = [
        jnp.array([-0.5, -0.5]).reshape(1, 2),
        jnp.array([-0.5, -0.25]).reshape(1, 2),
        jnp.array([-0.25, -0.75]).reshape(1, 2),
        jnp.array([-0.75, -0.5]).reshape(1, 2),
    ]
    J_bl = jnp.array([[1, 0], [0, 0]]).reshape(1, 2, 2)
    # top right corner
    xs_tr = [
        jnp.array([0.5, 0.5]).reshape(1, 2),
        jnp.array([0.5, 0.25]).reshape(1, 2),
        jnp.array([0.25, 0.5]).reshape(1, 2),
        jnp.array([0.25, 0.75]).reshape(1, 2),
        jnp.array([0.75, 0.5]).reshape(1, 2),
    ]
    J_tr = jnp.array([[0, 0], [0, 1]]).reshape(1, 2, 2)
    # points to the bottom right
    xs_br = [
        jnp.array([0.5, -0.5]).reshape(1, 2),
        jnp.array([0.5, -0.25]).reshape(1, 2),
        jnp.array([0.25, -0.5]).reshape(1, 2),
        jnp.array([0.25, -0.75]).reshape(1, 2),
        jnp.array([0.75, -0.5]).reshape(1, 2),
    ]
    J_br = jnp.zeros((1, 2, 2))
    # points to top left
    xs_tl = [
        jnp.array([-0.5, 0.5]).reshape(1, 2),
        jnp.array([-0.5, 0.25]).reshape(1, 2),
        jnp.array([-0.25, 0.5]).reshape(1, 2),
        jnp.array([-0.25, 0.75]).reshape(1, 2),
        jnp.array([-0.75, 0.5]).reshape(1, 2),
    ]
    J_tl = jnp.eye(2).reshape(1, 2, 2)

    if PLOT_RESULTS:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        quadrant = plt.Polygon(
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
        for color, points in zip(colors, point_sets):
            for x in points:
                ax.plot(x[0, 0], x[0, 1], "o", color=color)

        plt.grid(True)
        plt.show()

    lb = jnp.array([-jnp.inf, 0.0]).reshape((1, 2, 1))
    ub = jnp.array([0.0, jnp.inf]).reshape((1, 2, 1))
    box_constraint = BoxConstraint(lower_bound=lb, upper_bound=ub)
    projection_layer = Project(box_constraint=box_constraint)

    def fun_layer(x, v):
        return (projection_layer.call(x, 0.0) @ v).mean()

    e_1 = jnp.eye(2)[0]
    e_2 = jnp.eye(2)[1]

    def J_x(x):
        grad_e1 = (jax.grad(fun_layer, argnums=0)(x, e_1)).reshape(1, 1, 2)
        grad_e2 = (jax.grad(fun_layer, argnums=0)(x, e_2)).reshape(1, 1, 2)
        return jnp.concatenate((grad_e1, grad_e2), axis=1)

    # Check the Jacobian of the projection
    for xs, J_true in zip([xs_bl, xs_tr, xs_br, xs_tl], [J_bl, J_tr, J_br, J_tl]):
        for x in xs:
            J = J_x(x)
            assert jnp.allclose(
                J, J_true, atol=1e-4, rtol=1e-4
            ), f"J={J}, J_true={J_true} for x={x}"


# %%
# import cvxpy as cp
# import jaxopt
# def test_general_eq_ineq(method, seed, batch_size):
#     # %%
#     method = "pinv"
#     seed = 42
#     batch_size = 1
#     #
#     dim = 100
#     n_eq = 50
#     n_ineq = 40
#     key = jax.random.PRNGKey(seed)
#     key = jax.random.split(key, num=4)
#     # Generate equality constraints LHS
#     A = jax.random.normal(key[0], shape=(1, n_eq, dim))
#     # Generate inequality constraints LHS
#     C = jax.random.normal(key[1], shape=(1, n_ineq, dim))
#     # Compute RHS by solving feasibility problem
#     xfeas = cp.Variable(dim)
#     bfeas = cp.Variable(n_eq)
#     lfeas = cp.Variable(n_ineq)
#     ufeas = cp.Variable(n_ineq)
#     constraints = [
#         A[0, :, :] @ xfeas == bfeas,
#         lfeas <= C[0, :, :] @ xfeas,
#         C[0, :, :] @ xfeas <= ufeas,
#         -1 <= xfeas,
#         xfeas <= 1,
#     ]
#     objective = cp.Minimize(jnp.ones(shape=(dim)) @ xfeas)
#     problem = cp.Problem(objective=objective, constraints=constraints)
#     problem.solve()
#     # Extract RHS parameters
#     b = jnp.tile(jnp.array(bfeas.value).reshape((1, n_eq, 1)), (1, 1, 1))
#     lb = jnp.tile(jnp.array(lfeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))
#     ub = jnp.tile(jnp.array(ufeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))

#     # Define projection layer ingredients
#     eq_constraint = EqualityConstraint(A=A, b=b, method=method)
#     ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)
#     sigma = 1.0
#     omega = 1.0

#     # Projection layer with unrolling differentiation
#     projection_layer_unroll = Project(
#         eq_constraint=eq_constraint,
#         ineq_constraint=ineq_constraint,
#         unroll=True,
#         sigma=sigma,
#         omega=omega,
#     )

#     # Projection layer with implicit differentiation
#     projection_layer_impl = Project(
#         eq_constraint=eq_constraint,
#         ineq_constraint=ineq_constraint,
#         unroll=False,
#         sigma=sigma,
#         omega=omega,
#     )

#     # Point to be projected
#     x = jax.random.uniform(key[2], shape=(batch_size, dim), minval=-2, maxval=2)

#     # Compute the projection by solving QP
#     yqp = jnp.zeros(shape=(batch_size, dim))
#     for ii in range(batch_size):
#         yproj = cp.Variable(dim)
#         constraints = [
#             A[0, :, :] @ yproj == b[0, :, 0],
#             lb[0, :, 0] <= C[0, :, :] @ yproj,
#             C[0, :, :] @ yproj <= ub[0, :, 0],
#         ]
#         objective = cp.Minimize(cp.sum_squares(yproj - x[ii, :]))
#         problem_qp = cp.Problem(objective=objective, constraints=constraints)
#         problem_qp.solve()
#         yqp = yqp.at[ii, :].set(jnp.array(yproj.value).reshape((dim)))

#     # Check that the projection are computed correctly
#     n_iter = 200
#     y_unroll = projection_layer_unroll.call(x, n_iter=n_iter)[0]
#     y_impl = projection_layer_impl.call(x, n_iter=n_iter)[0]
#     assert jnp.allclose(y_unroll, yqp, atol=1e-4, rtol=1e-4)
#     assert jnp.allclose(y_impl, yqp, atol=1e-4, rtol=1e-4)

#     # %%
#     # Simple "loss" function as inner product
#     n_iter = 200
#     vec = jnp.array(jax.random.normal(key[3], shape=(dim, batch_size)))
#     def loss(x, v, unroll, n_iter_bwd, fpi):
#         if unroll:
#             return (projection_layer_unroll.call(x, n_iter=n_iter)[0] @ v).mean()
#         else:
#             return (projection_layer_impl.call(
#                 x,
#                 n_iter=n_iter,
#                 n_iter_bwd=n_iter_bwd,
#                 fpi=fpi,
#             )[0] @ v).mean()

#     grad_unroll = jax.grad(loss, argnums=0)(x, vec, True, n_iter_bwd=-1, fpi=True)
#     grad_fpi = jax.grad(loss, argnums=0)(x, vec, False, n_iter_bwd=200, fpi=True)
#     grad_ls = jax.grad(loss, argnums=0)(x, vec, False, n_iter_bwd=20, fpi=False)

#     assert jnp.allclose(grad_unroll, grad_fpi, atol=1e-4, rtol=1e-4)
#     assert jnp.allclose(grad_unroll, grad_ls, atol=1e-4, rtol=1e-4)
#     # %% Compute the gradient using a different method
#     # %% TODO: Compare with JAXopt, or cvxpylayers, or KKT computation
#     qp = jaxopt.OSQP(tol=1e-5)
#     Q = jnp.eye(dim)
#     Cqp = jnp.concatenate((C, -C), axis=1)
#     d = jnp.concatenate((ub, -lb), axis=1)
#     y_jaxopt, exact_project_vjp = jax.vjp(
#         lambda xx: qp.run(
#             params_obj=(Q, -xx), params_eq=(A[0, :, :], b[0, :, 0]),
#             params_ineq=(Cqp[0, :, :], d[0, :, 0])
#         ).params.primal,
#         x[0],
#     )
#     exact_project_vjp_jit = jax.jit(exact_project_vjp)
#     exact_project_vjp_jit(vec[:, 0])
#     # %%
#     print(exact_project_vjp_jit(vec[:, 0])[0])
#     print(grad_unroll[0,:10])
#     # %%
#     @jax.jit
#     def qp_proj_solver(infeas):
#         """Solve projection on polytope as a QP."""
#         sol = qp.run(params_obj=(Q, -infeas), params_eq=(A[0, :, :], b[0, :, 0]),
#              params_ineq=(Cqp[0, :, :], d[0, :, 0])).params

#         # Scalarize the output to compute the vjp
#         loss = jnp.dot(vec[:, 0], sol.primal)

#         return loss
#     # %%
#     print(f"Loss {qp_proj_solver(x[0])}")
#     # Compute the VJP with the cotvec
#     jaxopt_grad = jax.jit(lambda infeas: jax.grad(qp_proj_solver)(infeas))
#     jaxopt_grad(x[0])
#     # %%
#     from cvxpylayers.jax import CvxpyLayer
#     yproj = cp.Variable(dim)
#     xproj = cp.Parameter(dim)
#     constraints = [
#         A[0, :, :] @ yproj == b[0, :, 0],
#         lb[0, :, 0] <= C[0, :, :] @ yproj,
#         C[0, :, :] @ yproj <= ub[0, :, 0],
#     ]
#     objective = cp.Minimize(cp.sum_squares(yproj - xproj))
#     problem_qp = cp.Problem(objective=objective, constraints=constraints)
#     assert problem_qp.is_dpp()
#     x[ii, :]

#     cvxpylayer = CvxpyLayer(problem_qp, parameters=[xproj], variables=[yproj])
#     solution, = cvxpylayer(x[ii, :])
#     dcvxpylayer = jax.grad(lambda x: vec[:, 0] @ cvxpylayer(x)[0])
#     gradx = dcvxpylayer(x[ii, :])
#     # %%
#     print(jnp.linalg.norm(gradx - grad_unroll))
#     print(jnp.linalg.norm(gradx - exact_project_vjp_jit(vec[:, 0])[0]))
