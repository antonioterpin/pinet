"""Run HCNN on simple QP problem."""

# TODO: Add CLI.
# TODO: Add saving of results.
# TODO: Make the training batched.
# %%
import os
import timeit
from functools import partial

import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm

from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.project import Project

jax.config.update("jax_enable_x64", True)


# %% Compute optimal objectives on training set
def optimal_objectives(filename):
    """Compute the optimal of objectives for filename."""
    filename = (
        f"SimpleQP_seed{problem_seed}_var{problem_var}_ineq{problem_nineq}_"
        f"eq{problem_neq}_examples{problem_examples}.npz"
    )
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


# %%
class HardConstrainedMLP(nn.Module):
    """Simple MLP with hard constraints on the output."""

    project: Project

    def setup(self):
        """Setup for each NN call."""
        self.schedule = optax.linear_schedule(0.0, 0.0, 2000, 300)

    @nn.compact
    def __call__(self, x, step, n_iter=100):
        """Call the NN."""
        x = nn.Dense(200)(x)
        x = nn.relu(x)
        x = nn.Dense(200)(x)
        x = nn.relu(x)
        x = nn.Dense(self.project.dim)(x)
        alpha = self.schedule(step)
        x = self.project.call(x, interpolation_value=alpha, n_iter=n_iter)
        return x


# %% Import dataset
# Choose problem parameters
problem_seed = 42
problem_var = 100
problem_nineq = 50
problem_neq = 50
problem_examples = 10000
filename = (
    f"SimpleQP_seed{problem_seed}_var{problem_var}_ineq{problem_nineq}"
    f"_eq{problem_neq}_examples{problem_examples}.npz"
)
dataset_path = os.path.join(os.path.dirname(__file__), "datasets", filename)
data = jnp.load(dataset_path)
Q = data["Q"]
p = data["p"][0, :, :]
A = data["A"]
X = data["X"][:3000]
G = data["G"]
h = data["h"]
# Dimension of decision variable
Y_DIM = Q.shape[2]
# Dimension of parameter vector
X_DIM = X.shape[1]
N_SAMPLES = X.shape[0]
SEED = 0
LEARNING_RATE = 1e-2
# %% Setup constraint objects
eq_constraint = EqualityConstraint(A=A, b=X, method=None)
ineq_constraint = AffineInequalityConstraint(C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h))
projection_layer = Project(ineq_constraint=ineq_constraint, eq_constraint=eq_constraint)
projection_layer.setup()
# %%
model = HardConstrainedMLP(project=projection_layer)
params = model.init(jax.random.PRNGKey(SEED), X[:, :, 0], 0)
tx = optax.adam(LEARNING_RATE)
state = train_state.TrainState.create(
    apply_fn=model.apply, params=params["params"], tx=tx
)
# %% Get some predictions
predictions = state.apply_fn({"params": state.params}, X[:, :, 0], 0)


# %%
# Predictions is of shape (batch_size, Y_DIM) and Q is of shape (Y_DIM, Y_DIM)
def quadratic_form(prediction, Q):
    """Evaluate the quadratic objective."""
    return 0.5 * prediction.T @ Q @ prediction + p.T @ prediction


# Vectorize the quadratic form computation over the batch dimension
batched_quadratic_form = jax.vmap(quadratic_form, in_axes=[0, None])


# %% Setup the MLP training routine
@partial(jax.jit, static_argnames=["n_iter"])
def train_step(state, x_batch, step, n_iter):
    """Run a single training step."""

    def loss_fn(params):
        predictions = state.apply_fn({"params": params}, x_batch, step, n_iter)
        return batched_quadratic_form(predictions, Q).mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return loss, state.apply_gradients(grads=grads)


# %% Train the MLP
N_EPOCHS = 100
PLOT_TRAINING = True
n_iter_train = 50
losses = []
eqcvs = []
ineqcvs = []

for step in (pbar := tqdm(range(N_EPOCHS))):
    loss, state = train_step(state, X[:, :, 0], step, n_iter_train)
    losses.append(loss)
    pbar.set_description(f"Loss: {loss:.5f}")

    if step % 10 == 0:
        predictions = state.apply_fn(
            {"params": state.params}, X[:, :, 0], 10000000, n_iter=n_iter_train
        )
        eqcv = jnp.abs(A[0] @ predictions.reshape(-1, Y_DIM, 1) - X).max()
        ineqcv = jnp.maximum(G[0] @ predictions.reshape(-1, Y_DIM, 1) - h, 0).max()
        eqcvs.append(eqcv)
        ineqcvs.append(ineqcv)
        pbar.set_postfix({"eqcv": f"{eqcv:.5f}", "ineqcv": f"{ineqcv:.5f}"})

# Plot the results
if PLOT_TRAINING:
    opt_objective = optimal_objectives(filename)[: X.shape[0]].mean()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.axhline(
        y=opt_objective,
        color="r",
        linestyle="-",
        linewidth=2,
        label="Optimal Objective",
    )
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(eqcvs, label="Equality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Equality Violation")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(ineqcvs, label="Inequality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Inequality Violation")
    plt.legend()

    plt.tight_layout()
    plt.show()
# %% Evaluate training performance
n_iter_test = 100
scheduler_step = 10000000
predictions = state.apply_fn(
    {"params": state.params}, X[:, :, 0], scheduler_step, n_iter=n_iter_test
)
# Average objective function
obj_fun_train = batched_quadratic_form(predictions, Q).mean()
opt_objectives = optimal_objectives(filename)[: X.shape[0]]
# Average and max equality constraint violation
eq_viol_train_mean = jnp.abs(
    A[0].reshape(1, problem_neq, Y_DIM) @ predictions.reshape(X.shape[0], Y_DIM, 1) - X
).mean()
eq_viol_train_max = jnp.abs(
    A[0].reshape(1, problem_neq, Y_DIM) @ predictions.reshape(X.shape[0], Y_DIM, 1) - X
).max()
# Average and max inequality constraint violation
ineq_viol_train_mean = jnp.maximum(
    G[0].reshape(1, problem_nineq, Y_DIM) @ predictions.reshape(X.shape[0], Y_DIM, 1)
    - h,
    0,
).mean()
ineq_viol_train_max = jnp.maximum(
    G[0].reshape(1, problem_nineq, Y_DIM) @ predictions.reshape(X.shape[0], Y_DIM, 1)
    - h,
    0,
).max()
# Computation time
evals = 10
time_train_evaluation = (
    timeit.timeit(
        lambda: state.apply_fn(
            {"params": state.params}, X[:, :, 0], scheduler_step, n_iter=n_iter_test
        ).block_until_ready(),
        number=evals,
    )
    / evals
)
# Print the results
print("=========== Training performance ===========")
print("Mean objective                : ", f"{obj_fun_train:.5f}")
print(
    "Mean|Max equality violation   : ",
    f"{eq_viol_train_mean:.5f}",
    "|",
    f"{eq_viol_train_max:.5f}",
)
print(
    "Mean|Max inequality violation : ",
    f"{ineq_viol_train_mean:.5f}",
    "|",
    f"{ineq_viol_train_max:.5f}",
)
print("Time for evaluation [s]       : ", f"{time_train_evaluation:.5f}")
print("Optimal mean objective        : ", f"{opt_objectives.mean():.5f}")

# %% Solve some training individual problem
problem_idx = 0
ycp = cp.Variable(Y_DIM)
constraints = [A[0, :, :] @ ycp == X[problem_idx, :, 0], G[0, :, :] @ ycp <= h[0, :, 0]]
objective = cp.Minimize(0.5 * cp.quad_form(ycp, Q[0, :, :]) + p.T @ ycp)
problem = cp.Problem(objective=objective, constraints=constraints)
problem.solve(solver=cp.OSQP)
print("OSQP")
print(problem.value)
print(jnp.abs(A[0] @ ycp.value - X[problem_idx, :, 0]).max())
print(jnp.maximum(G[0] @ ycp.value - h[0, :, 0], 0).max())
print("HCNN")
predictions = state.apply_fn({"params": state.params}, X[:, :, 0], 10000000, n_iter=100)
print(quadratic_form(predictions[problem_idx, :], Q).item())
print(jnp.abs(A[0] @ predictions[problem_idx, :] - X[problem_idx, :, 0]).max())
print(jnp.maximum(G[0] @ predictions[problem_idx, :] - h[0, :, 0], 0).max())
# %% Do testing
Xtest = data["X"].reshape(10000, 50, 1)[-1000:]
eq_constraint_test = EqualityConstraint(A=A, b=Xtest, method=None)
ineq_constraint_test = AffineInequalityConstraint(
    C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h)
)
projection_layer_test = Project(
    ineq_constraint=ineq_constraint_test, eq_constraint=eq_constraint_test
)
projection_layer_test.setup()
model.project = projection_layer_test
# %%
n_iter_test = 50
predictions = state.apply_fn(
    {"params": state.params}, Xtest[:, :, 0], scheduler_step, n_iter=n_iter_test
)
# Average objective function
obj_fun_test = batched_quadratic_form(predictions, Q).mean()
# Compute optimal objective on test set
opt_objectives = optimal_objectives(filename)[-Xtest.shape[0] :]
# Average and max equality constraint violation
eq_viol_test_mean = jnp.abs(
    A[0].reshape(1, problem_neq, Y_DIM) @ predictions.reshape(Xtest.shape[0], Y_DIM, 1)
    - Xtest
).mean()
eq_viol_test_max = jnp.abs(
    A[0].reshape(1, problem_neq, Y_DIM) @ predictions.reshape(Xtest.shape[0], Y_DIM, 1)
    - Xtest
).max()
# Average and max inequality constraint violation
ineq_viol_test_mean = jnp.maximum(
    G[0].reshape(1, problem_nineq, Y_DIM)
    @ predictions.reshape(Xtest.shape[0], Y_DIM, 1)
    - h,
    0,
).mean()
ineq_viol_test_max = jnp.maximum(
    G[0].reshape(1, problem_nineq, Y_DIM)
    @ predictions.reshape(Xtest.shape[0], Y_DIM, 1)
    - h,
    0,
).max()
# Computation time
evals = 10
time_test_evaluation = (
    timeit.timeit(
        lambda: state.apply_fn(
            {"params": state.params}, Xtest[:, :, 0], scheduler_step, n_iter=n_iter_test
        ).block_until_ready(),
        number=evals,
    )
    / evals
)
# Print the results
print("=========== Testing performance ===========")
print("Mean objective                : ", f"{obj_fun_test:.5f}")
print(
    "Mean|Max equality violation   : ",
    f"{eq_viol_test_mean:.5f}",
    "|",
    f"{eq_viol_test_max:.5f}",
)
print(
    "Mean|Max inequality violation : ",
    f"{ineq_viol_test_mean:.5f}",
    "|",
    f"{ineq_viol_test_max:.5f}",
)
print("Time for evaluation [s]       : ", f"{time_test_evaluation:.5f}")
print("Optimal mean objective        : ", f"{opt_objectives.mean():.5f}")
# %% Solve some test problems
problem_idx = 0
ycp = cp.Variable(Y_DIM)
constraints = [
    A[0, :, :] @ ycp == Xtest[problem_idx, :, 0],
    G[0, :, :] @ ycp <= h[0, :, 0],
]
objective = cp.Minimize(0.5 * cp.quad_form(ycp, Q[0, :, :]) + p.T @ ycp)
problem = cp.Problem(objective=objective, constraints=constraints)
problem.solve(solver=cp.OSQP, verbose=False)
print("OSQP")
print(problem.value)
print(jnp.abs(A[0] @ ycp.value - Xtest[problem_idx, :, 0]).max())
print(jnp.maximum(G[0] @ ycp.value - h[0, :, 0], 0).max())
print("HCNN")
predictions = state.apply_fn(
    {"params": state.params}, Xtest[:, :, 0], 10000000, n_iter=50
)
print(quadratic_form(predictions[problem_idx, :], Q).item())
print(jnp.abs(A[0] @ predictions[problem_idx, :] - Xtest[problem_idx, :, 0]).max())
print(jnp.maximum(G[0] @ predictions[problem_idx, :] - h[0, :, 0], 0).max())
