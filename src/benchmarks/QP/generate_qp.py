"""Generate simple QP problem.

Using the approach from the DC3 paper:
https://arxiv.org/pdf/2104.12225
"""

import os
from typing import cast

import cvxpy as cp
import jax
import jax.numpy as jnp
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


def solve_problems(
    q_mat: jnp.ndarray,
    p: jnp.ndarray,
    a_dyn: jnp.ndarray,
    x_data: jnp.ndarray,
    g_mat: jnp.ndarray,
    h: jnp.ndarray,
    convex: bool = True,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the optimal solutions for problem instances.

    Args:
        q_mat: Quadratic term of the objective function.
        p: Linear term of the objective function.
        a_dyn: Coefficient matrix for equality constraints.
        x_data: Input data for equality constraints.
        g_mat: Coefficient matrix for inequality constraints.
        h: Right-hand side of the inequality constraints.
        convex: Whether the problem is convex or not.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]:
            A tuple containing the objectives and optimal solutions.

    Raises:
        NotImplementedError: If the problem is not convex.
        RuntimeError: If OSQP does not return a solution for a problem instance.
    """
    # Dimension of decision variable
    y_dim = q_mat.shape[2]
    n_samples = x_data.shape[0]
    if convex:
        print("Solving problem instances")
        objectives = jnp.zeros(n_samples)
        ystar = jnp.zeros((n_samples, y_dim, 1))
        for problem_idx in tqdm(range(n_samples)):
            ycp = cp.Variable(y_dim)
            constraints = cast(
                list[cp.Constraint],
                [
                    a_dyn[0, :, :] @ ycp == x_data[problem_idx, :, 0],
                    g_mat[0, :, :] @ ycp <= h[0, :, 0],
                ],
            )
            objective = cp.Minimize(0.5 * cp.quad_form(ycp, q_mat[0, :, :]) + p.T @ ycp)
            problem = cp.Problem(objective=objective, constraints=constraints)
            problem.solve(solver=cp.OSQP)
            if ycp.value is None:
                raise RuntimeError(
                    f"OSQP did not return a solution for problem {problem_idx}."
                )
            objectives = objectives.at[problem_idx].set(problem.value)
            ystar = ystar.at[problem_idx, :, :].set(jnp.expand_dims(ycp.value, axis=1))
    else:
        raise NotImplementedError

    return objectives, ystar


if __name__ == "__main__":
    # Save dataset flag
    SAVE_DATASET = True

    # Parameters setup
    seed = 42
    NUM_VAR = 100
    NUM_INEQ = 50
    NUM_EQ = 50
    NUM_EXAMPLES = 200
    CONVEX = True

    # Setup keys
    key = jax.random.PRNGKey(seed)
    key = jax.random.split(key, 5)

    # Generate matrices
    q_mat = jnp.expand_dims(
        jnp.diag(jax.random.uniform(key[0], shape=(NUM_VAR,), minval=0.0, maxval=1.0)),
        axis=0,
    )
    p = jax.random.uniform(key[1], shape=(1, NUM_VAR, 1), minval=0.0, maxval=1.0)
    a_dyn = jax.random.normal(key[2], shape=(1, NUM_EQ, NUM_VAR))
    x_data = jax.random.uniform(
        key[3], shape=(NUM_EXAMPLES, NUM_EQ, 1), minval=-1.0, maxval=1.0
    )
    g_mat = jax.random.normal(key[4], shape=(1, NUM_INEQ, NUM_VAR))
    h = jnp.expand_dims(
        jnp.sum(jnp.abs(g_mat @ jnp.linalg.pinv(a_dyn[0])), axis=1), axis=2
    )

    if SAVE_DATASET:
        # Create the datasets directory if it doesn't exist
        datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")

        # Solve all the problem instances
        objectives, ystar = solve_problems(
            q_mat, p[0, :, :], a_dyn, x_data, g_mat, h, CONVEX
        )

        # Define the filename and save the dataset
        if CONVEX:
            filename = os.path.join(
                datasets_dir,
                (
                    f"SimpleQP_seed{seed}_var{NUM_VAR}_ineq{NUM_INEQ}"
                    f"_eq{NUM_EQ}_examples{NUM_EXAMPLES}.npz"
                ),
            )
        else:
            filename = os.path.join(
                datasets_dir,
                (
                    f"Simple_nonconvex_seed{seed}_var{NUM_VAR}_ineq{NUM_INEQ}"
                    f"_eq{NUM_EQ}_examples{NUM_EXAMPLES}.npz"
                ),
            )
        jnp.savez(
            filename,
            q_mat=q_mat,
            p=p,
            a_dyn=a_dyn,
            x_data=x_data,
            g_mat=g_mat,
            h=h,
            objectives=objectives,
            ystar=ystar,
        )
