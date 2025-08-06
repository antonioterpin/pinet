"""Test projection layer and vjp, with equilibration."""

from itertools import product

import cvxpy as cp
import jax
import pytest
from jax import numpy as jnp

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    EqualityConstraint,
    EqualityInputs,
    Inputs,
    Project,
)

jax.config.update("jax_enable_x64", True)

SEEDS = [24, 42]
BATCH_SIZE = [1, 5]


@pytest.mark.parametrize("seed, batch_size", product(SEEDS, BATCH_SIZE))
def test_general_eq_ineq(seed, batch_size):
    method = "pinv"
    dim = 100
    n_eq = 50
    n_ineq = 40
    n_box = 15
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, num=5)
    # Generate equality constraints LHS
    A = jax.random.normal(key[0], shape=(1, n_eq, dim))
    # Generate inequality constraints LHS
    C = jax.random.normal(key[1], shape=(1, n_ineq, dim))
    # Randomly generate mask for box constraints
    indices = jnp.concatenate([jnp.ones(n_box), jnp.zeros(dim - n_box)])
    mask = jax.random.permutation(key[2], indices).astype(bool)
    # Compute RHS by solving feasibility problem
    xfeas = cp.Variable(dim * batch_size)
    bfeas = cp.Variable(n_eq * batch_size)
    lfeas = cp.Variable(n_ineq)
    ufeas = cp.Variable(n_ineq)
    lboxfeas = cp.Variable(n_box)
    uboxfeas = cp.Variable(n_box)
    constraints = [
        -1 <= lfeas,
        lfeas <= 1,
        -1 <= ufeas,
        ufeas <= 1,
        -1 <= lboxfeas,
        lboxfeas <= 1,
        -1 <= uboxfeas,
        uboxfeas <= 1,
    ]
    for ii in range(batch_size):
        constraints += [
            A[0, :, :] @ xfeas[ii * dim : (ii + 1) * dim]
            == bfeas[ii * n_eq : (ii + 1) * n_eq],
            lfeas <= C[0, :, :] @ xfeas[ii * dim : (ii + 1) * dim],
            C[0, :, :] @ xfeas[ii * dim : (ii + 1) * dim] <= ufeas,
            lboxfeas <= xfeas[ii * dim : (ii + 1) * dim][mask],
            xfeas[ii * dim : (ii + 1) * dim][mask] <= uboxfeas,
            -2 <= xfeas,
            xfeas <= 2,
        ]
    objective = cp.Minimize(jnp.ones(shape=(dim * batch_size)) @ xfeas)
    problem = cp.Problem(objective=objective, constraints=constraints)
    problem.solve(verbose=True)
    # Extract RHS parameters
    b = jnp.tile(jnp.array(bfeas.value).reshape((batch_size, n_eq, 1)), (1, 1, 1))
    lb = jnp.tile(jnp.array(lfeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))
    ub = jnp.tile(jnp.array(ufeas.value).reshape((1, n_ineq, 1)), (1, 1, 1))
    lbox = jnp.tile(jnp.array(lboxfeas.value).reshape((1, n_box, 1)), (1, 1, 1))
    ubox = jnp.tile(jnp.array(uboxfeas.value).reshape((1, n_box, 1)), (1, 1, 1))
    # Define projection layer ingredients
    for var_b in [False, True]:
        eq_constraint = EqualityConstraint(A=A, b=b, method=method, var_b=var_b)
        ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)
        box_constraint = BoxConstraint(lower_bound=lbox, upper_bound=ubox, mask=mask)

        # Hyperparameters
        sigma = 0.1
        omega = 1.7
        sigma_equil = 0.01

        # Projection layer with unrolling no equilibration
        pl_unroll = Project(
            eq_constraint=eq_constraint,
            ineq_constraint=ineq_constraint,
            box_constraint=box_constraint,
            unroll=True,
            equilibrate={
                "max_iter": 0,
                "tol": 1e-3,
                "ord": 2.0,
                "col_scaling": False,
                "update_mode": "Gauss",
                "safeguard": False,
            },
        )

        # Projection layer with unrolling plus equilibration
        pl_unroll_equil = Project(
            eq_constraint=eq_constraint,
            ineq_constraint=ineq_constraint,
            box_constraint=box_constraint,
            unroll=True,
            equilibrate={
                "max_iter": 25,
                "tol": 1e-3,
                "ord": 2.0,
                "col_scaling": False,
                "update_mode": "Gauss",
                "safeguard": False,
            },
        )

        # Projection layer with implicit differentiation
        pl_impl_equil = Project(
            eq_constraint=eq_constraint,
            ineq_constraint=ineq_constraint,
            box_constraint=box_constraint,
            unroll=False,
            equilibrate={
                "max_iter": 25,
                "tol": 1e-3,
                "ord": 2.0,
                "col_scaling": False,
                "update_mode": "Gauss",
                "safeguard": False,
            },
        )
        # Point to be projected
        x = jax.random.uniform(key[3], shape=(batch_size, dim), minval=-2, maxval=2)

        # Compute the projection by solving QP
        yqp = jnp.zeros(shape=(batch_size, dim))
        for ii in range(batch_size):
            yproj = cp.Variable(dim)
            constraints = [
                A[0, :, :] @ yproj == b[ii, :, 0],
                lb[0, :, 0] <= C[0, :, :] @ yproj,
                C[0, :, :] @ yproj <= ub[0, :, 0],
                lbox[0, :, 0] <= yproj[mask],
                yproj[mask] <= ubox[0, :, 0],
            ]
            objective = cp.Minimize(cp.sum_squares(yproj - x[ii, :]))
            problem_qp = cp.Problem(objective=objective, constraints=constraints)
            problem_qp.solve()
            yqp = yqp.at[ii, :].set(jnp.array(yproj.value).reshape((dim)))

        # Check that the projection is computed correctly
        n_iter = 1000
        if var_b:
            inp = Inputs(x=x, eq=EqualityInputs(b=b))
        else:
            inp = Inputs(x=x)
        y_unroll = pl_unroll.call(
            pl_unroll.get_init(inp), inp, n_iter=n_iter, sigma=sigma, omega=omega
        )[0]
        y_impl = pl_unroll_equil.call(
            pl_unroll_equil.get_init(inp),
            inp,
            n_iter=n_iter,
            sigma=sigma_equil,
            omega=omega,
        )[0]
        y_impl_equil = pl_impl_equil.call(
            pl_impl_equil.get_init(inp),
            inp,
            n_iter=n_iter,
            sigma=sigma_equil,
            omega=omega,
        )[0]
        assert jnp.allclose(y_unroll, yqp, atol=1e-4, rtol=1e-4)
        assert jnp.allclose(y_impl, yqp, atol=1e-4, rtol=1e-4)
        assert jnp.allclose(y_impl_equil, yqp, atol=1e-4, rtol=1e-4)

        # Check that the VJP is computed correctly
        # Compare with loop unrolling
        # Simple "loss" function as inner product
        n_iter = 1000
        vec = jnp.array(jax.random.normal(key[4], shape=(dim, batch_size)))

        def loss(x, v, mode, n_iter_bwd, fpi):
            if var_b:
                inp = Inputs(x=x, eq=EqualityInputs(b=b))
            else:
                inp = Inputs(x=x)
            if mode == "unroll":
                return (
                    pl_unroll.call(
                        pl_unroll.get_init(inp),
                        inp,
                        n_iter=n_iter,
                        sigma=sigma,
                        omega=omega,
                    )[0]
                    @ v
                ).mean()
            elif mode == "unroll_equil":
                return (
                    pl_unroll_equil.call(
                        pl_unroll_equil.get_init(inp),
                        inp,
                        n_iter=n_iter,
                        sigma=sigma_equil,
                        omega=omega,
                    )[0]
                    @ v
                ).mean()
            elif mode == "impl_equil":
                return (
                    pl_impl_equil.call(
                        pl_impl_equil.get_init(inp),
                        inp,
                        n_iter=n_iter,
                        sigma=sigma_equil,
                        omega=omega,
                        n_iter_bwd=n_iter_bwd,
                        fpi=fpi,
                    )[0]
                    @ v
                ).mean()

        grad_unroll = jax.grad(loss, argnums=0)(
            x, vec, "unroll", n_iter_bwd=-1, fpi=True
        )
        grad_unroll_equil = jax.grad(loss, argnums=0)(
            x, vec, "unroll_equil", n_iter_bwd=-1, fpi=True
        )
        grad_impl_equil = jax.grad(loss, argnums=0)(
            x, vec, "impl_equil", n_iter_bwd=100, fpi=False
        )

        assert jnp.allclose(grad_unroll, grad_unroll_equil, atol=1e-4, rtol=1e-4)
        assert jnp.allclose(grad_unroll, grad_impl_equil, atol=1e-4, rtol=1e-4)
