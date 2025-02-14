"""Generate simple QP problem.

Using the approach from the DC3 paper:
https://arxiv.org/pdf/2104.12225
"""

import os

import cvxpy as cp
import jax
import jax.numpy as jnp
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)


def optimal_objectives(filename):
    """Compute the optimal of objectives for filename."""
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", filename)

    data = jnp.load(dataset_path)
    Q = data["Q"]
    p = data["p"][0, :, :]
    A = data["A"]
    X = data["X"]
    G = data["G"]
    h = data["h"]
    # Dimension of decision variable
    Y_DIM = Q.shape[2]
    N_SAMPLES = X.shape[0]
    solution_path = os.path.join(
        os.path.dirname(__file__), "datasets", filename.replace(".npz", "_solution.npz")
    )
    if os.path.exists(solution_path):
        solution_data = jnp.load(solution_path)
        objectives = solution_data["objectives"]
    else:
        print("Solving problem instances")
        objectives = jnp.zeros(N_SAMPLES)
        for problem_idx in tqdm(range(N_SAMPLES)):
            ycp = cp.Variable(Y_DIM)
            constraints = [
                A[0, :, :] @ ycp == X[problem_idx, :, 0],
                G[0, :, :] @ ycp <= h[0, :, 0],
            ]
            objective = cp.Minimize(0.5 * cp.quad_form(ycp, Q[0, :, :]) + p.T @ ycp)
            problem = cp.Problem(objective=objective, constraints=constraints)
            problem.solve(solver=cp.OSQP)
            objectives = objectives.at[problem_idx].set(problem.value)

        jnp.savez(solution_path, objectives=objectives)

    return objectives


if __name__ == "__main__":
    # Save dataset flag
    SAVE_DATASET = False

    # Parameters setup
    SEED = 42
    NUM_VAR = 1000
    NUM_INEQ = 500
    NUM_EQ = 500
    NUM_EXAMPLES = 2000

    # Setup keys
    key = jax.random.PRNGKey(SEED)
    key = jax.random.split(key, 5)

    # Generate matrices
    Q = jnp.expand_dims(
        jnp.diag(jax.random.uniform(key[0], shape=(NUM_VAR,), minval=0.0, maxval=1.0)),
        axis=0,
    )
    p = jax.random.uniform(key[1], shape=(1, NUM_VAR, 1), minval=0.0, maxval=1.0)
    A = jax.random.normal(key[2], shape=(1, NUM_EQ, NUM_VAR))
    X = jax.random.uniform(
        key[3], shape=(NUM_EXAMPLES, NUM_EQ, 1), minval=-1.0, maxval=1.0
    )
    G = jax.random.normal(key[4], shape=(1, NUM_INEQ, NUM_VAR))
    h = jnp.expand_dims(jnp.sum(jnp.abs(G @ jnp.linalg.pinv(A[0])), axis=1), axis=2)

    if SAVE_DATASET:
        # Create the datasets directory if it doesn't exist
        datasets_dir = os.path.join(os.path.dirname(__file__), "datasets")

        # Define the filename and save the dataset
        filename = os.path.join(
            datasets_dir,
            (
                f"SimpleQP_seed{SEED}_var{NUM_VAR}_ineq{NUM_INEQ}"
                f"_eq{NUM_EQ}_examples{NUM_EXAMPLES}.npz"
            ),
        )
        jnp.savez(filename, Q=Q, p=p, A=A, X=X, G=G, h=h)
        # Save also the optimal objectives for each problem
        _ = optimal_objectives(filename)
