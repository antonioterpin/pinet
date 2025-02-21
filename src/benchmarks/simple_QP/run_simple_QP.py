"""Run HCNN on simple QP problem."""

# TODO: Add CLI.
# %%
import datetime
import os
import pathlib
import time
import timeit
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import yaml
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm

from benchmarks.simple_QP.load_simple_QP import (
    SimpleQPDataset,
    create_dataloaders,
    dc3_dataloader,
)
from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.project import Project

jax.config.update("jax_enable_x64", True)


# %% Import hyperparameters
# Load hyperparameters from a yaml file
def load_hyperparameters(file_path: str) -> dict:
    """Load hyperparameters for HCNN."""
    with open(file_path, "r") as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters


# Configs path
config_path = (
    pathlib.Path(__file__).parent.parent.parent.resolve() / "configs" / "simple_QP.yaml"
)
# Example usage
hyperparameters = load_hyperparameters(config_path)


# %% Setup CLI
# Inputs:
# Problem parameters: Seed, Variables, # Ineq, # Eq, # Examples
# Use DC3 dataset or not
# Use convex problem or not
# To save the results or not
# HCNN hyperparameters: defaults to default.yaml, or provide your own yaml
#   This includes train/test iterations, learning rate, scheduler?
#       Epochs,
# %%
class HardConstrainedMLP(nn.Module):
    """Simple MLP with hard constraints on the output."""

    project: Project

    def setup(self):
        """Setup for each NN call."""
        self.schedule = optax.linear_schedule(0.0, 0.0, 2000, 300)

    # TODO: Try adding batch norm and dropout as in the DC3 paper.
    # A quick try with batch norm generated slightly worse results.
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


# %% Setup problem
# Use existing DC3 Dataset or own dataset
use_DC3_dataset = True
use_convex = False
SETUP_RERS = 10
# Import dataset
problem_seed = 42
problem_var = 100
problem_nineq = 50
problem_neq = 50
problem_examples = 10000
if not use_DC3_dataset:
    # Choose problem parameters
    if use_convex:
        filename = (
            f"SimpleQP_seed{problem_seed}_var{problem_var}_ineq{problem_nineq}"
            f"_eq{problem_neq}_examples{problem_examples}.npz"
        )
    else:
        raise NotImplementedError()
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", filename)

    QPDataset = SimpleQPDataset(dataset_path)
    train_loader, valid_loader, test_loader = create_dataloaders(
        dataset_path, batch_size=2000, val_split=0.1, test_split=0.1
    )
    Q, p, A, G, h = QPDataset.const
    p = p[0, :, :]
    X = QPDataset.X
else:
    # Choose the filename here
    if use_convex:
        raise NotImplementedError()
        filename = (
            f"dc3_random_dataset_var{problem_var}_ineq{problem_nineq}"
            f"_eq{problem_neq}_ex{problem_examples}"
        )
    else:
        filename = (
            f"dc3_random_nonconvex_dataset_var{problem_var}_ineq{problem_nineq}"
            f"_eq{problem_neq}_ex{problem_examples}"
        )
    filename_train = filename + "train.npz"
    dataset_path_train = os.path.join(
        os.path.dirname(__file__), "datasets", filename_train
    )
    filename_valid = filename + "valid.npz"
    dataset_path_valid = os.path.join(
        os.path.dirname(__file__), "datasets", filename_valid
    )
    filename_test = filename + "test.npz"
    dataset_path_test = os.path.join(
        os.path.dirname(__file__), "datasets", filename_test
    )
    train_loader = dc3_dataloader(dataset_path_train, batch_size=2000)
    valid_loader = dc3_dataloader(dataset_path_valid, batch_size=2000, shuffle=False)
    test_loader = dc3_dataloader(dataset_path_test, batch_size=2000, shuffle=False)
    Q, p, A, G, h = train_loader.dataset.const
    p = p[0, :, :]
    X = train_loader.dataset.X
# Dimension of decision variable
Y_DIM = Q.shape[2]
# Dimension of parameter vector
SEED = 0
LEARNING_RATE = hyperparameters["learning_rate"]
eq_constraint = EqualityConstraint(A=A, b=X, method=None, var_b=True)
ineq_constraint = AffineInequalityConstraint(C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h))
projection_layer = Project(ineq_constraint=ineq_constraint, eq_constraint=eq_constraint)
start = time.time()
if SETUP_RERS > 0:
    for _ in range(SETUP_RERS):
        eq_constraint = EqualityConstraint(A=A, b=X, method=None, var_b=True)
        ineq_constraint = AffineInequalityConstraint(
            C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h)
        )
        projection_layer = Project(
            ineq_constraint=ineq_constraint, eq_constraint=eq_constraint
        )
    setup_time = (time.time() - start) / SETUP_RERS

    print(f"Time to create constraints: {setup_time:.5f} seconds")
else:
    setup_time = -1
# %%
model = HardConstrainedMLP(project=projection_layer)
params = model.init(jax.random.PRNGKey(SEED), x=X[:2, :, 0], b=X[:2], step=0, n_iter=2)
tx = optax.adam(LEARNING_RATE)
state = train_state.TrainState.create(
    apply_fn=model.apply, params=params["params"], tx=tx
)


# %%
# Predictions is of shape (batch_size, Y_DIM) and Q is of shape (Y_DIM, Y_DIM)
def quadratic_form(prediction):
    """Evaluate the quadratic objective."""
    return 0.5 * prediction.T @ Q @ prediction + p.T @ prediction


def quadratic_form_sine(prediction):
    """Evaluate the quadratic objective plus sine."""
    return 0.5 * prediction.T @ Q @ prediction + p.T @ jnp.sin(prediction)


if use_convex:
    objective_function = quadratic_form
else:
    objective_function = quadratic_form_sine


# Vectorize the quadratic form computation over the batch dimension
batched_objective = jax.vmap(objective_function, in_axes=[0])


# To be used if we include a constraint violation penalty in the objective
def penalty_form(prediction):
    """Penaly for violating inequality constraints."""
    return jnp.maximum(
        G[0].reshape(problem_nineq, Y_DIM) @ prediction - h,
        0,
    ).max()


batched_penalty_form = jax.vmap(penalty_form, in_axes=[0])


# %% Setup the MLP training routine
@partial(jax.jit, static_argnames=["n_iter"])
def train_step(state, x_batch, b_batch, step, n_iter):
    """Run a single training step."""

    def loss_fn(params):
        predictions = state.apply_fn({"params": params}, x_batch, b_batch, step, n_iter)
        return batched_objective(predictions).mean()

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return loss, state.apply_gradients(grads=grads)


for batch in train_loader:
    X_batch, _ = batch
start = time.time()
compiled_fn = train_step.lower(state, X_batch[:, :, 0], X_batch, 0, 100).compile()
# Note this also includes the time for one iteration
compilation_time = time.time() - start

print(f"Compilation time: {compilation_time:.5f} seconds")
# %% Train the MLP
N_EPOCHS = hyperparameters["n_epochs"]
PLOT_TRAINING = True
start = time.time()
n_iter_train = hyperparameters["n_iter_train"]
n_iter_test = hyperparameters["n_iter_test"]
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
            loss = batched_objective(predictions).mean()
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
training_time = time.time() - start
print(f"Training time: {training_time:.5f} seconds")
# %%
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
n_iter_test = hyperparameters["n_iter_test"]
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
obj_fun_valid = batched_objective(predictions).mean()
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
if not use_DC3_dataset:
    # import cvxpy as cp
    # ycp = cp.Variable(Y_DIM)
    # constraints = [A[0, :, :] @ ycp == X_valid[problem_idx, :, 0],
    #                G[0, :, :] @ ycp <= h[0, :, 0]]
    # objective = cp.Minimize(0.5 * cp.quad_form(ycp, Q[0, :, :]) + p.T @ ycp)
    # problem = cp.Problem(objective=objective, constraints=constraints)
    # problem.solve(solver=cp.OSQP)
    # print("OSQP")
    # print(problem.value)
    # print(jnp.abs(A[0] @ ycp.value - X_valid[problem_idx, :, 0]).max())
    # print(jnp.maximum(G[0] @ ycp.value - h[0, :, 0], 0).max())
    objective_val = valid_loader.dataset.dataset.objectives[
        valid_loader.dataset.indices[problem_idx]
    ]
    eqcv_val = jnp.abs(
        A[0]
        @ valid_loader.dataset.dataset.Ystar[valid_loader.dataset.indices[problem_idx]]
        - valid_loader.dataset.dataset.X[
            valid_loader.dataset.indices[problem_idx], :, :
        ]
    ).max()
    ineqcv_val = jnp.maximum(
        G[0]
        @ valid_loader.dataset.dataset.Ystar[valid_loader.dataset.indices[problem_idx]]
        - h[0, :, :],
        0,
    ).max()

    print("Stored Optimal Solution")
    print(f"Objective: \t{objective_val:.5e}")
    print(f"Eq. cv:    \t{eqcv_val:.5e}")
    print(f"Ineq. cv:  \t{ineqcv_val:.5e}")
else:
    objective_val = valid_loader.dataset.objectives[problem_idx].item()
    eqcv_val = jnp.abs(
        A[0] @ valid_loader.dataset.Ystar[problem_idx]
        - valid_loader.dataset.X[problem_idx, :, :]
    ).max()
    ineqcv_val = jnp.maximum(
        G[0] @ valid_loader.dataset.Ystar[problem_idx] - h[0, :, :],
        0,
    ).max()

    print("Stored Optimal Solution")
    print(f"Objective:  \t{objective_val:.5e}")
    print(f"Eq. cv:     \t{eqcv_val:.5e}")
    print(f"Ineq. cv:   \t{ineqcv_val:.5e}")
# Predictions from validation set
objective_val_hcnn = objective_function(predictions[problem_idx, :]).item()
eqcv_val_hcnn = jnp.abs(
    A[0] @ predictions[problem_idx, :] - X_valid[problem_idx, :, 0]
).max()
ineqcv_val_hcnn = jnp.maximum(G[0] @ predictions[problem_idx, :] - h[0, :, 0], 0).max()
print("HCNN")
print(f"Objective:  \t{objective_val_hcnn:.5e}")
print(f"Eq. cv:     \t{eqcv_val_hcnn:.5e}")
print(f"Ineq. cv:   \t{ineqcv_val_hcnn:.5e}")
# %%
n_iter_test = hyperparameters["n_iter_test"]
for X_test, obj_test in test_loader:
    predictions = state.apply_fn(
        {"params": state.params},
        X_test[:, :, 0],
        X_test,
        scheduler_step,
        n_iter=n_iter_test,
    )
opt_test_loss = obj_test.mean()
# Average objective function
obj_fun_test = batched_objective(predictions)
obj_fun_test_mean = obj_fun_test.mean()
# Average and max equality constraint violation
eq_viol_test = jnp.abs(
    A[0].reshape(1, problem_neq, Y_DIM) @ predictions.reshape(X_test.shape[0], Y_DIM, 1)
    - X_test
)
eq_viol_test_mean = eq_viol_test.mean()
eq_viol_test_max = eq_viol_test.max()
# Average and max inequality constraint violation
ineq_viol_test = jnp.maximum(
    G[0].reshape(1, problem_nineq, Y_DIM)
    @ predictions.reshape(X_test.shape[0], Y_DIM, 1)
    - h,
    0,
).mean()
ineq_viol_test_mean = ineq_viol_test.mean()
ineq_viol_test_max = ineq_viol_test.max()
# Computation time
evals = 5
time_test_evaluation = (
    timeit.timeit(
        lambda: state.apply_fn(
            {"params": state.params},
            X_test[:, :, 0],
            X_test,
            scheduler_step,
            n_iter=n_iter_test,
        ).block_until_ready(),
        number=evals,
    )
    / evals
)
# Print the results
print("=========== Testing performance ===========")
print("Mean objective                : ", f"{obj_fun_test_mean:.5f}")
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
if not use_DC3_dataset:
    # ycp = cp.Variable(Y_DIM)
    # constraints = [
    #     A[0, :, :] @ ycp == X_test[problem_idx, :, 0],
    #     G[0, :, :] @ ycp <= h[0, :, 0],
    # ]
    # objective = cp.Minimize(0.5 * cp.quad_form(ycp, Q[0, :, :]) + p.T @ ycp)
    # problem = cp.Problem(objective=objective, constraints=constraints)
    # problem.solve(solver=cp.OSQP, verbose=False)
    # print("OSQP")
    # print(problem.value)
    # print(jnp.abs(A[0] @ ycp.value - X_test[problem_idx, :, 0]).max())
    # print(jnp.maximum(G[0] @ ycp.value - h[0, :, 0], 0).max())
    objective_val = test_loader.dataset.dataset.objectives[
        test_loader.dataset.indices[problem_idx]
    ]
    eqcv_val = jnp.abs(
        A[0]
        @ test_loader.dataset.dataset.Ystar[test_loader.dataset.indices[problem_idx]]
        - test_loader.dataset.dataset.X[test_loader.dataset.indices[problem_idx], :, :]
    ).max()
    ineqcv_val = jnp.maximum(
        G[0]
        @ test_loader.dataset.dataset.Ystar[test_loader.dataset.indices[problem_idx]]
        - h[0, :, :],
        0,
    ).max()

    print("Stored Optimal Solution")
    print(f"Objective: \t{objective_val:.5e}")
    print(f"Eq. cv:    \t{eqcv_val:.5e}")
    print(f"Ineq. cv:  \t{ineqcv_val:.5e}")
else:
    objective_val = test_loader.dataset.objectives[problem_idx].item()
    eqcv_val = jnp.abs(
        A[0] @ test_loader.dataset.Ystar[problem_idx]
        - test_loader.dataset.X[problem_idx, :, :]
    ).max()
    ineqcv_val = jnp.maximum(
        G[0] @ test_loader.dataset.Ystar[problem_idx] - h[0, :, :], 0
    ).max()

    print("Stored Optimal Solution")
    print(f"Objective:  \t{objective_val:.5e}")
    print(f"Eq. cv:     \t{eqcv_val:.5e}")
    print(f"Ineq. cv:   \t{ineqcv_val:.5e}")

objective_val_hcnn = objective_function(predictions[problem_idx, :]).item()
eqcv_val_hcnn = jnp.abs(
    A[0] @ predictions[problem_idx, :] - X_test[problem_idx, :, 0]
).max()
ineqcv_val_hcnn = jnp.maximum(G[0] @ predictions[problem_idx, :] - h[0, :, 0], 0).max()

print("HCNN")
print(f"Objective:  \t{objective_val_hcnn:.5e}")
print(f"Eq. cv:     \t{eqcv_val_hcnn:.5e}")
print(f"Ineq. cv:   \t{ineqcv_val_hcnn:.5e}")
# %% Saving of overall results
# Total training time = Precomputation + Jit + Real Training
# On the test set:
#   Evaluation time
#   Mean + Max constraint violation (equality + inequality)
#   Distribution of constraint violations?
#   Objective Values -> All of them
#       From this we could extract: average RS, max RS, Distribution of RS
#       (RS = Relative Suboptimality)
#   Optimal Objective Values
# Add the datetime to the name of the results to ensure uniqueness
SAVE_RESULTS = True
if SAVE_RESULTS:
    # Setup results path
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_results = "results_" + current_timestamp + "_" + filename
    results_path = dataset_path = os.path.join(
        os.path.dirname(__file__), "results", filename_results
    )

    jnp.savez(
        file=results_path,
        inference_time=time_test_evaluation,
        setup_time=setup_time,
        compilation_time=compilation_time,
        training_time=training_time,
        eq_viol_test=eq_viol_test,
        ineq_viol_test=ineq_viol_test,
        obj_fun_test=obj_fun_test,
        opt_obj_test=obj_test,
    )
