"""Solver module for SOC benchmarks."""

from typing import Callable

import cvxpy as cp
import jax.numpy as jnp
import numpy as np
import scipy
import scs
from tqdm import tqdm


def setup_scs(
    A_np: jnp.ndarray,
    b_sample: jnp.ndarray,
    c_sample: jnp.ndarray,
    n: int,
    m: int,
    eps: float = 1e-3,
    verbose: bool = False,
):
    """Set up an SCS solver for Second-Order Cone Programming (SOCP) problems.

    Args:
        A_np (jnp.ndarray): Constraint matrix of shape (m, n).
        b_sample (jnp.ndarray): Right-hand side vector for constraints, shape (m,).
        c_sample (jnp.ndarray): Objective function coefficients, shape (n,).
        n (int): Number of optimization variables.
        m (int): Number of equality constraints.
        eps (float, optional): Solver tolerance for both absolute and relative error.
            Defaults to 1e-3.
        verbose (bool, optional): Whether to enable verbose solver output.
            Defaults to False.

    Returns:
        Callable: A solve function that takes (b_sample, c_sample) and returns
                 (x_solution, s_slack, solve_time) where:
                 - x_solution: Optimal primal variables (n, 1)
                 - s_slack: Optimal slack variables (m, 1)
                 - solve_time: Total setup and solve time in seconds
    """
    # Convert to SCS format
    A_scs = scipy.sparse.csc_matrix(
        A_np
    )  # Constraint matrix for equality constraints (sparse)

    # Problem data for this instance
    b_sample = np.asarray(b_sample).ravel()  # shape (m,)
    c_sample = np.asarray(c_sample).ravel()  # shape (n,)

    # Data dictionary for SCS
    data = {
        "A": A_scs,
        "b": b_sample,
        "c": c_sample,
    }

    # Cone dictionary - equality constraints and SOC
    cone = {
        "z": m - 1,  # Number of zero cone constraints (equality)
        "q": [1],  # SOC constraint of size 1 (last component)
    }

    # Setup solver
    solver = scs.SCS(data, cone, eps_abs=eps, eps_rel=eps, verbose=verbose)

    def solve(b_sample: jnp.ndarray, c_sample: jnp.ndarray):
        solver.update(b=np.asarray(b_sample).ravel(), c=np.asarray(c_sample).ravel())
        sol = solver.solve()
        if sol["info"]["status"] != "solved":
            raise RuntimeError(f"Problem status: {sol['info']['status']}")
        return (
            sol["x"].reshape(n, 1),
            sol["s"].reshape(m, 1),
            1e-3 * (sol["info"]["setup_time"] + sol["info"]["solve_time"]),
        )

    return solve


def setup_cvxpy_parametric(
    A_np: jnp.ndarray,
    b_sample: jnp.ndarray,
    c_sample: jnp.ndarray,
    n: int,
    m: int,
    eps: float = 1e-3,
    verbose: bool = False,
):
    """Set up a parametric CVXPY solver for Second-Order Cone Programming (SOCP) problems.

    Args:
        A_np (jnp.ndarray): Constraint matrix of shape (m, n).
        b_sample (jnp.ndarray): Right-hand side vector for constraints, shape (m,).
        c_sample (jnp.ndarray): Objective function coefficients, shape (n,).
        n (int): Number of optimization variables.
        m (int): Number of equality constraints.
        eps (float, optional): Solver tolerance for both absolute and relative error.
            Defaults to 1e-3.
        verbose (bool, optional): Whether to enable verbose solver output.
            Defaults to False.

    Returns:
        Callable: A solve function that takes (b_sample, c_sample) and returns
                 (x_solution, s_slack, solve_time) where:
                 - x_solution: Optimal primal variables (n, 1)
                 - s_slack: Optimal slack variables (m, 1)
                 - solve_time: Total setup and solve time in seconds
    """
    x_var = cp.Variable(n)
    s_var = cp.Variable(m)
    b_par = cp.Parameter(m)
    c_par = cp.Parameter(n)

    constraints = [A_np @ x_var + s_var == b_par, cp.SOC(s_var[-1], s_var[:-1])]

    problem = cp.Problem(cp.Minimize(c_par @ x_var), constraints)

    def solve(b_sample: jnp.ndarray, c_sample: jnp.ndarray):
        b_par.value = np.asarray(b_sample).ravel()
        c_par.value = np.asarray(c_sample).ravel()
        problem.solve(solver=cp.SCS, eps_abs=eps, eps_rel=eps, verbose=verbose)
        if x_var.value is None:
            raise RuntimeError(f"Problem status: {problem.status}")
        return (
            x_var.value.reshape(n, 1),
            s_var.value.reshape(m, 1),
            problem.solver_stats.setup_time
            + problem.solver_stats.solve_time
            + problem.compilation_time,
        )

    return solve


def setup_cvxpy(
    A_np: jnp.ndarray,
    b_sample: jnp.ndarray,
    c_sample: jnp.ndarray,
    n: int,
    m: int,
    eps: float = 1e-3,
    verbose: bool = False,
):
    """Set up a non-parametric CVXPY solver for SOCPs.

    Args:
        A_np (jnp.ndarray): Constraint matrix of shape (m, n).
        b_sample (jnp.ndarray): Right-hand side vector for constraints, shape (m,).
        c_sample (jnp.ndarray): Objective function coefficients, shape (n,).
        n (int): Number of optimization variables.
        m (int): Number of equality constraints.
        eps (float, optional): Solver tolerance for both absolute and relative error.
            Defaults to 1e-3.
        verbose (bool, optional): Whether to enable verbose solver output.
            Defaults to False.

    Returns:
        Callable: A solve function that takes (b_sample, c_sample) and returns
                 (x_solution, s_slack, solve_time) where:
                 - x_solution: Optimal primal variables (n, 1)
                 - s_slack: Optimal slack variables (m, 1)
                 - solve_time: Total setup and solve time in seconds
    """

    def solve(b_sample: jnp.ndarray, c_sample: jnp.ndarray):
        b_sample = np.asarray(b_sample).ravel()
        c_sample = np.asarray(c_sample).ravel()
        x_var = cp.Variable(n)
        s_var = cp.Variable(m)
        constraints = [A_np @ x_var + s_var == b_sample, cp.SOC(s_var[-1], s_var[:-1])]
        problem = cp.Problem(cp.Minimize(c_sample @ x_var), constraints)
        problem.solve(solver=cp.SCS, eps_abs=eps, eps_rel=eps, verbose=verbose)
        if x_var.value is None:
            raise RuntimeError(f"Problem status: {problem.status}")
        return (
            x_var.value.reshape(n, 1),
            s_var.value.reshape(m, 1),
            problem.solver_stats.setup_time
            + problem.solver_stats.solve_time
            + problem.compilation_time,
        )

    return solve


def _evaluate_solver(
    solve: Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, float]],
    b_batch: jnp.ndarray,
    c_batch: jnp.ndarray,
    use_tqdm: bool = True,
):
    """Evaluate a solver function on a batch of problems.

    This is a helper that applies a solver function to multiple problem instances,
    collecting the solutions and timing information for each solve.

    Args:
        solve (Callable): A solver function that takes (b_sample, c_sample) and returns
                         (x_solution, s_slack, solve_time).
        b_batch (jnp.ndarray): Batch of right-hand side vectors, shape (B, m) where
                              B is the batch size and m is the number of constraints.
        c_batch (jnp.ndarray): Batch of objective coefficients, shape (B, n) where
                              B is the batch size and n is the number of variables.
        use_tqdm (bool, optional): Whether to use tqdm for progress display.
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - x_cvxpy (jnp.ndarray): Optimal primal solutions for all problems,
                shape (B, n, 1)
            - s_cvxpy (jnp.ndarray): Optimal slack variables for all problems,
                shape (B, m, 1)
            - times (jnp.ndarray): Solve times for all problems, shape (B,)
    """
    B = b_batch.shape[0]
    x_sol = []
    s_sol = []
    times = []
    if use_tqdm:
        iterator = tqdm(range(B))
    else:
        iterator = range(B)
    for i in iterator:
        x_i, s_i, time_i = solve(b_batch[i], c_batch[i])
        x_sol.append(x_i)
        s_sol.append(s_i)
        times.append(time_i)
    n = x_sol[0].shape[0]
    m = s_sol[0].shape[0]
    x_cvxpy = jnp.asarray(x_sol).reshape(B, n, 1)
    s_cvxpy = jnp.asarray(s_sol).reshape(B, m, 1)
    times = jnp.array(times)
    return x_cvxpy, s_cvxpy, times


def evaluate_solver(
    A_np: jnp.ndarray,
    b_batch: jnp.ndarray,
    c_batch: jnp.ndarray,
    n: int,
    m: int,
    eps: float = 1e-3,
    verbose: bool = False,
    method: str = "cvxpy_parametric",
    use_tqdm: bool = True,
):
    """Evaluate a specified solver method on a batch of SOCPs.

    Args:
        A_np (jnp.ndarray): Constraint matrix of shape (m, n).
        b_sample (jnp.ndarray): Right-hand side vector for constraints, shape (m,).
        c_sample (jnp.ndarray): Objective function coefficients, shape (n,).
        n (int): Number of optimization variables.
        m (int): Number of equality constraints.
        eps (float, optional): Solver tolerance for both absolute and relative error.
            Defaults to 1e-3.
        verbose (bool, optional): Whether to enable verbose solver output.
            Defaults to False.
        method (str, optional): Solver method to use. Options are:
                    - "scs": Direct SCS solver interface
                    - "cvxpy_parametric": CVXPY with parametric problem formulation
                    - "cvxpy": CVXPY with problem rebuilt each time
                    Defaults to "cvxpy_parametric".
        use_tqdm (bool, optional): Whether to use tqdm for progress display.
            Defaults to True.

    Returns:
        tuple: A tuple containing:
            - x_solutions (jnp.ndarray): Optimal primal solutions for all problems,
                shape (B, n, 1)
            - s_solutions (jnp.ndarray): Optimal slack variables for all problems,
                shape (B, m, 1)
            - solve_times (jnp.ndarray): Solve times for all problems, shape (B,)

    Raises:
        ValueError: If the specified method is not one of the supported options.
    """
    setups = {
        "scs": setup_scs,
        "cvxpy_parametric": setup_cvxpy_parametric,
        "cvxpy": setup_cvxpy,
    }
    solve = setups[method](
        A_np=A_np,
        b_sample=b_batch[0],
        c_sample=c_batch[0],
        n=n,
        m=m,
        eps=eps,
        verbose=verbose,
    )
    return _evaluate_solver(solve, b_batch, c_batch, use_tqdm=use_tqdm)
