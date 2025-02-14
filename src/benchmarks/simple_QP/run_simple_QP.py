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

from benchmarks.simple_QP.load_simple_QP import SimpleQPDataset, create_dataloaders
from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.project import Project

jax.config.update("jax_enable_x64", True)


# %%
class HardConstrainedMLP(nn.Module):
    """Simple MLP with hard constraints on the output."""

    project: Project

    def setup(self):
        """Setup for each NN call."""
        self.schedule = optax.linear_schedule(0.0, 0.0, 2000, 300)

    @nn.compact
    def __call__(self, x, b, step, n_iter=100):
        """Call the NN."""
        x = nn.Dense(200)(x)
        x = nn.relu(x)
        x = nn.Dense(200)(x)
        x = nn.relu(x)
        x = nn.Dense(self.project.dim)(x)
        alpha = self.schedule(step)
        x = self.project.call(x, b, interpolation_value=alpha, n_iter=n_iter)
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

QPDataset = SimpleQPDataset(dataset_path)
train_loader, valid_loader, test_loader = create_dataloaders(
    dataset_path, batch_size=2000, val_split=0.1, test_split=0.1
)
Q, p, A, G, h = QPDataset.const
p = p[0, :, :]
X = QPDataset.X
# Dimension of decision variable
Y_DIM = Q.shape[2]
# Dimension of parameter vector
SEED = 0
LEARNING_RATE = 1e-2
# %% Setup constraint objects
eq_constraint = EqualityConstraint(A=A, b=X, method=None, var_b=True)
ineq_constraint = AffineInequalityConstraint(C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h))
projection_layer = Project(ineq_constraint=ineq_constraint, eq_constraint=eq_constraint)
# %%
model = HardConstrainedMLP(project=projection_layer)
params = model.init(jax.random.PRNGKey(SEED), X[:2, :, 0], X[:2], 0, 2)
tx = optax.adam(LEARNING_RATE)
state = train_state.TrainState.create(
    apply_fn=model.apply, params=params["params"], tx=tx
)


# %%
# Predictions is of shape (batch_size, Y_DIM) and Q is of shape (Y_DIM, Y_DIM)
def quadratic_form(prediction, Q):
    """Evaluate the quadratic objective."""
    return 0.5 * prediction.T @ Q @ prediction + p.T @ prediction


# Vectorize the quadratic form computation over the batch dimension
batched_quadratic_form = jax.vmap(quadratic_form, in_axes=[0, None])


# %% Setup the MLP training routine
@partial(jax.jit, static_argnames=["n_iter"])
def train_step(state, x_batch, b_batch, step, n_iter):
    """Run a single training step."""

    def loss_fn(params):
        predictions = state.apply_fn({"params": params}, x_batch, b_batch, step, n_iter)
        return batched_quadratic_form(predictions, Q).mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return loss, state.apply_gradients(grads=grads)


# %% Train the MLP
N_EPOCHS = 100
PLOT_TRAINING = True
n_iter_train = 50
n_iter_test = 50
trainig_losses = []
validation_losses = []
eqcvs = []
ineqcvs = []
for step in (pbar := tqdm(range(N_EPOCHS))):
    epoch_loss = []
    for batch in train_loader:
        X_batch, _ = batch
        loss, state = train_step(state, X_batch[:, :, 0], X_batch, step, n_iter_train)
        epoch_loss.append(loss)
    pbar.set_description(f"Train Loss: {jnp.array(epoch_loss).mean():.5f}")
    trainig_losses.append(jnp.array(epoch_loss).mean())

    if step % 5 == 0:
        for X_valid, _ in valid_loader:
            predictions = state.apply_fn(
                {"params": state.params},
                X_valid[:, :, 0],
                X_valid,
                10000000,
                n_iter=n_iter_test,
            )
            loss = batched_quadratic_form(predictions, Q).mean()
            eqcv = jnp.abs(A[0] @ predictions.reshape(-1, Y_DIM, 1) - X_valid).max()
            ineqcv = jnp.maximum(G[0] @ predictions.reshape(-1, Y_DIM, 1) - h, 0).max()
            eqcvs.append(eqcv)
            ineqcvs.append(ineqcv)
            validation_losses.append(loss)
            pbar.set_postfix(
                {
                    "eqcv": f"{eqcv:.5f}",
                    "ineqcv": f"{ineqcv:.5f}",
                    "Valid. Loss:": f"{loss:.5f}",
                }
            )

# Plot the results
if PLOT_TRAINING:
    opt_train_loss = []
    for batch in train_loader:
        _, obj_batch = batch
        opt_train_loss.append(obj_batch)
    opt_train_loss = jnp.concatenate(opt_train_loss, axis=0).mean()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.plot(trainig_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.axhline(
        y=opt_train_loss,
        color="r",
        linestyle="-",
        linewidth=2,
        label="Optimal Training Objective",
    )
    plt.legend()

    opt_valid_loss = []
    for batch in valid_loader:
        opt_valid_loss.append(obj_batch)
    opt_valid_loss = jnp.array(opt_valid_loss).mean()
    plt.subplot(1, 4, 2)
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.axhline(
        y=opt_valid_loss,
        color="r",
        linestyle="-",
        linewidth=2,
        label="Optimal Validation Objective",
    )
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(eqcvs, label="Equality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Equality Violation")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(ineqcvs, label="Inequality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Inequality Violation")
    plt.legend()

    plt.tight_layout()
    plt.show()
# %% Evaluate validation performance
n_iter_test = 50
scheduler_step = 10000000
# Only one batch for validation set
for X_valid, obj_valid in valid_loader:
    predictions = state.apply_fn(
        {"params": state.params},
        X_valid[:, :, 0],
        X_valid,
        scheduler_step,
        n_iter=n_iter_test,
    )
opt_valid_loss = obj_valid.mean()
# Average objective function
obj_fun_valid = batched_quadratic_form(predictions, Q).mean()
# Average and max equality constraint violation
eq_viol_valid_mean = jnp.abs(
    A[0].reshape(1, problem_neq, Y_DIM)
    @ predictions.reshape(X_valid.shape[0], Y_DIM, 1)
    - X_valid
).mean()
eq_viol_valid_max = jnp.abs(
    A[0].reshape(1, problem_neq, Y_DIM)
    @ predictions.reshape(X_valid.shape[0], Y_DIM, 1)
    - X_valid
).max()
# Average and max inequality constraint violation
ineq_viol_valid_mean = jnp.maximum(
    G[0].reshape(1, problem_nineq, Y_DIM)
    @ predictions.reshape(X_valid.shape[0], Y_DIM, 1)
    - h,
    0,
).mean()
ineq_viol_valid_max = jnp.maximum(
    G[0].reshape(1, problem_nineq, Y_DIM)
    @ predictions.reshape(X_valid.shape[0], Y_DIM, 1)
    - h,
    0,
).max()
# Computation time
evals = 5
time_valid_evaluation = (
    timeit.timeit(
        lambda: state.apply_fn(
            {"params": state.params},
            X_valid[:, :, 0],
            X_valid,
            scheduler_step,
            n_iter=n_iter_test,
        ).block_until_ready(),
        number=evals,
    )
    / evals
)
# Print the results
print("=========== Training performance ===========")
print("Mean objective                : ", f"{obj_fun_valid:.5f}")
print(
    "Mean|Max equality violation   : ",
    f"{eq_viol_valid_mean:.5f}",
    "|",
    f"{eq_viol_valid_max:.5f}",
)
print(
    "Mean|Max inequality violation : ",
    f"{ineq_viol_valid_mean:.5f}",
    "|",
    f"{ineq_viol_valid_max:.5f}",
)
print("Time for evaluation [s]       : ", f"{time_valid_evaluation:.5f}")
print("Optimal mean objective        : ", f"{opt_valid_loss:.5f}")

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
predictions = state.apply_fn(
    {"params": state.params}, X[:, :, 0], X, 10000000, n_iter=n_iter_test
)
print(quadratic_form(predictions[problem_idx, :], Q).item())
print(jnp.abs(A[0] @ predictions[problem_idx, :] - X[problem_idx, :, 0]).max())
print(jnp.maximum(G[0] @ predictions[problem_idx, :] - h[0, :, 0], 0).max())
# %%
n_iter_test = 50
for Xtest, obj_test in test_loader:
    predictions = state.apply_fn(
        {"params": state.params},
        Xtest[:, :, 0],
        Xtest,
        scheduler_step,
        n_iter=n_iter_test,
    )
opt_test_loss = obj_test.mean()
# Average objective function
obj_fun_test = batched_quadratic_form(predictions, Q).mean()
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
evals = 5
time_test_evaluation = (
    timeit.timeit(
        lambda: state.apply_fn(
            {"params": state.params},
            Xtest[:, :, 0],
            Xtest,
            scheduler_step,
            n_iter=n_iter_test,
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
print("Optimal mean objective        : ", f"{opt_test_loss:.5f}")
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
