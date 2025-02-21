"""Run HCNN on simple QP problem."""

# TODO: Enable saving/serializing the trained network when saving results.
import argparse
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
def load_yaml(file_path: str) -> dict:
    """Load hyperparameters for HCNN."""
    with open(file_path, "r") as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters


def plotting(
    train_loader, valid_loader, trainig_losses, validation_losses, eqcvs, ineqcvs
):
    """Plot training curves."""
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


# %% Evaluation function to clean up code
def evaluate_hcnn(
    loader,
    state,
    n_iter,
    batched_objective,
    A,
    G,
    h,
    prefix,
    time_evals=5,
    print_res=True,
):
    """Evaluate the performance of the HCNN."""
    # This assumes the loader handles all the data in one batch.
    for X, obj in loader:
        predictions = state.apply_fn(
            {"params": state.params},
            X[:, :, 0],
            X,
            100000,
            n_iter=n_iter,
        )
    opt_obj = obj.mean()
    # HCNN objective
    hcnn_obj = batched_objective(predictions)
    hcnn_obj_mean = hcnn_obj.mean()
    # Average and max equality constraint violation
    eq_cv = jnp.abs(
        A[0].reshape(1, A.shape[1], A.shape[2])
        @ predictions.reshape(X.shape[0], A.shape[2], 1)
        - X
    )
    eq_cv_mean = eq_cv.mean()
    eq_cv_max = eq_cv.max()
    # Average and max inequality constraint violation
    ineq_cv = jnp.maximum(
        G[0].reshape(1, G.shape[1], G.shape[2])
        @ predictions.reshape(X.shape[0], G.shape[2], 1)
        - h,
        0,
    )
    ineq_cv_mean = ineq_cv.mean()
    ineq_cv_max = ineq_cv.max()
    # Computation time
    eval_time = (
        timeit.timeit(
            lambda: state.apply_fn(
                {"params": state.params},
                X[:, :, 0],
                X,
                100000,
                n_iter=n_iter,
            ).block_until_ready(),
            number=time_evals,
        )
        / time_evals
    )
    if print_res:
        print(f"=========== {prefix} performance ===========")
        print("Mean objective                : ", f"{hcnn_obj_mean:.5f}")
        print(
            "Mean|Max equality violation   : ",
            f"{eq_cv_mean:.5f}",
            "|",
            f"{eq_cv_max:.5f}",
        )
        print(
            "Mean|Max inequality violation : ",
            f"{ineq_cv_mean:.5f}",
            "|",
            f"{ineq_cv_max:.5f}",
        )
        print("Time for evaluation [s]       : ", f"{eval_time:.5f}")
        print("Optimal mean objective        : ", f"{opt_obj:.5f}")

    return (opt_obj, hcnn_obj, eq_cv, ineq_cv, eval_time)


# %% Evaluate individual instance
def evaluate_instance(
    problem_idx,
    loader,
    state,
    n_iter,
    use_DC3_dataset,
    batched_objective,
    A,
    G,
    h,
    prefix,
):
    """Evaluate performance on single problem instance."""
    # Evaluate HCNN solution
    # This assumes the loader handles all the data in one batch.
    for X, obj in loader:
        pass

    predictions = state.apply_fn(
        {"params": state.params},
        X[problem_idx, :, 0].reshape((1, X.shape[1])),
        X[problem_idx].reshape((1, X.shape[1], 1)),
        100000,
        n_iter=n_iter,
    )

    objective_val_hcnn = batched_objective(predictions).item()
    eqcv_val_hcnn = jnp.abs(
        A[0] @ predictions.reshape(A.shape[2]) - X[problem_idx, :, 0]
    ).max()
    ineqcv_val_hcnn = jnp.maximum(
        G[0] @ predictions.reshape(G.shape[2]) - h[0, :, 0], 0
    ).max()
    print(f"=========== {prefix} individual performance ===========")
    print("HCNN")
    print(f"Objective:  \t{objective_val_hcnn:.5e}")
    print(f"Eq. cv:     \t{eqcv_val_hcnn:.5e}")
    print(f"Ineq. cv:   \t{ineqcv_val_hcnn:.5e}")

    # Evaluate optimal solution
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
        objective_val = loader.dataset.dataset.objectives[
            loader.dataset.indices[problem_idx]
        ]
        eqcv_val = jnp.abs(
            A[0] @ loader.dataset.dataset.Ystar[loader.dataset.indices[problem_idx]]
            - loader.dataset.dataset.X[loader.dataset.indices[problem_idx], :, :]
        ).max()
        ineqcv_val = jnp.maximum(
            G[0] @ loader.dataset.dataset.Ystar[loader.dataset.indices[problem_idx]]
            - h[0, :, :],
            0,
        ).max()

    else:
        objective_val = loader.dataset.objectives[problem_idx].item()
        eqcv_val = jnp.abs(
            A[0] @ loader.dataset.Ystar[problem_idx]
            - loader.dataset.X[problem_idx, :, :]
        ).max()
        ineqcv_val = jnp.maximum(
            G[0] @ loader.dataset.Ystar[problem_idx] - h[0, :, :], 0
        ).max()

    print("Optimal Solution")
    print(f"Objective:  \t{objective_val:.5e}")
    print(f"Eq. cv:     \t{eqcv_val:.5e}")
    print(f"Ineq. cv:   \t{ineqcv_val:.5e}")


def load_data(
    use_DC3_dataset,
    use_convex,
    problem_seed,
    problem_var,
    problem_nineq,
    problem_neq,
    problem_examples,
):
    """Load problem data."""
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
        valid_loader = dc3_dataloader(
            dataset_path_valid, batch_size=2000, shuffle=False
        )
        test_loader = dc3_dataloader(dataset_path_test, batch_size=2000, shuffle=False)
        Q, p, A, G, h = train_loader.dataset.const
        p = p[0, :, :]
        X = train_loader.dataset.X

    return (filename, Q, p, A, G, h, X, train_loader, valid_loader, test_loader)


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


def main(
    use_DC3_dataset,
    use_convex,
    problem_seed,
    problem_var,
    problem_nineq,
    problem_neq,
    problem_examples,
    config_path,
    SEED,
    PLOT_TRAINING,
    SAVE_RESULTS,
):
    """Main for running simple QP benchmarks."""
    # Load hyperparameter configuration
    hyperparameters = load_yaml(config_path)

    # Load problem data
    (filename, Q, p, A, G, h, X, train_loader, valid_loader, test_loader) = load_data(
        use_DC3_dataset,
        use_convex,
        problem_seed,
        problem_var,
        problem_nineq,
        problem_neq,
        problem_examples,
    )

    # Dimension of decision variable
    Y_DIM = Q.shape[2]
    # Dimension of parameter vector
    LEARNING_RATE = hyperparameters["learning_rate"]

    # Setup the projection layer
    eq_constraint = EqualityConstraint(A=A, b=X, method=None, var_b=True)
    ineq_constraint = AffineInequalityConstraint(
        C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h)
    )
    projection_layer = Project(
        ineq_constraint=ineq_constraint,
        eq_constraint=eq_constraint,
        sigma=hyperparameters["sigma"],
        omega=hyperparameters["omega"],
    )

    # Measure setup time
    SETUP_REPS = 10
    start = time.time()
    if SETUP_REPS > 0:
        for _ in range(SETUP_REPS):
            eq_constraint = EqualityConstraint(A=A, b=X, method=None, var_b=True)
            ineq_constraint = AffineInequalityConstraint(
                C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h)
            )
            projection_layer = Project(
                ineq_constraint=ineq_constraint,
                eq_constraint=eq_constraint,
                sigma=hyperparameters["sigma"],
                omega=hyperparameters["omega"],
            )
        setup_time = (time.time() - start) / SETUP_REPS

        print(f"Time to create constraints: {setup_time:.5f} seconds")
    else:
        setup_time = -1

    # Define HCNN model
    model = HardConstrainedMLP(project=projection_layer)
    params = model.init(
        jax.random.PRNGKey(SEED), x=X[:2, :, 0], b=X[:2], step=0, n_iter=2
    )
    tx = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params["params"], tx=tx
    )

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
    # def penalty_form(prediction):
    #     """Penaly for violating inequality constraints."""
    #     return jnp.maximum(
    #         G[0].reshape(problem_nineq, Y_DIM) @ prediction - h,
    #         0,
    #     ).max()

    # batched_penalty_form = jax.vmap(penalty_form, in_axes=[0])

    # Setup the MLP training routine
    @partial(jax.jit, static_argnames=["n_iter"])
    def train_step(state, x_batch, b_batch, step, n_iter):
        """Run a single training step."""

        def loss_fn(params):
            predictions = state.apply_fn(
                {"params": params}, x_batch, b_batch, step, n_iter
            )
            return batched_objective(predictions).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return loss, state.apply_gradients(grads=grads)

    # Measure compilation time
    for batch in train_loader:
        X_batch, _ = batch
    start = time.time()
    _ = train_step.lower(state, X_batch[:, :, 0], X_batch, 0, 100).compile()
    # Note this also includes the time for one iteration
    compilation_time = time.time() - start

    print(f"Compilation time: {compilation_time:.5f} seconds")

    # Train the MLP
    N_EPOCHS = hyperparameters["n_epochs"]
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
            loss, state = train_step(
                state, X_batch[:, :, 0], X_batch, step, n_iter_train
            )
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
                ineqcv = jnp.maximum(
                    G[0] @ predictions.reshape(-1, Y_DIM, 1) - h, 0
                ).max()
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

    # Plot the results
    if PLOT_TRAINING:
        plotting(
            train_loader,
            valid_loader,
            trainig_losses,
            validation_losses,
            eqcvs,
            ineqcvs,
        )

    # Evaluate validation performance
    _ = evaluate_hcnn(
        loader=valid_loader,
        state=state,
        n_iter=hyperparameters["n_iter_test"],
        batched_objective=batched_objective,
        prefix="Validation",
        A=A,
        G=G,
        h=h,
    )
    # Solve some validation individual problem
    problem_idx = 4
    evaluate_instance(
        problem_idx,
        valid_loader,
        state,
        hyperparameters["n_iter_test"],
        use_DC3_dataset,
        batched_objective,
        A,
        G,
        h,
        "Validation",
    )

    (obj_test, obj_fun_test, eq_viol_test, ineq_viol_test, time_test_evaluation) = (
        evaluate_hcnn(
            loader=test_loader,
            state=state,
            n_iter=hyperparameters["n_iter_test"],
            batched_objective=batched_objective,
            prefix="Testing",
            A=A,
            G=G,
            h=h,
        )
    )
    # Solve some test problems
    problem_idx = 0
    evaluate_instance(
        problem_idx,
        test_loader,
        state,
        hyperparameters["n_iter_test"],
        use_DC3_dataset,
        batched_objective,
        A,
        G,
        h,
        "Testing",
    )

    # Saving of overall results
    if SAVE_RESULTS:
        # Setup results path
        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_results = "results_" + current_timestamp + "_" + filename
        results_path = os.path.join(
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

    return state


# %%
if __name__ == "__main__":

    def parse_args():
        """Parse CLI arguments."""
        parser = argparse.ArgumentParser(
            description="Run HCNN on simple QP or nonconvex problem."
        )
        parser.add_argument(
            "--id",
            type=str,
            required=True,
            help="Yaml file specifying the dataset, see the `ids` folder.",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="simple_QP.yaml",
            help="Configuration file for HCNN hyperparameters.",
        )
        parser.add_argument(
            "--seed", type=int, default=42, help="Seed for training HCNN."
        )
        parser.add_argument(
            "--plot_training", type=bool, default=True, help="Plot training curves."
        )
        parser.add_argument(
            "--save_results", action="store_true", help="Save the results."
        )
        parser.add_argument(
            "--no-save-results",
            action="store_false",
            dest="save_results",
            help="Don't save the results.",
        )
        parser.set_defaults(save_results=True)
        return parser.parse_args()

    args = parse_args()
    # Load the yaml file
    idpath = pathlib.Path(__file__).parent.resolve() / "ids" / args.id
    dataset = load_yaml(idpath)
    # Use existing DC3 Dataset or own dataset
    use_DC3_dataset = dataset["use_DC3_dataset"]
    use_convex = dataset["use_convex"]
    # Import dataset
    problem_seed = dataset["problem_seed"]
    problem_var = dataset["problem_var"]
    problem_nineq = dataset["problem_nineq"]
    problem_neq = dataset["problem_neq"]
    problem_examples = dataset["problem_examples"]
    # Configs path
    config_path = (
        pathlib.Path(__file__).parent.parent.parent.resolve() / "configs" / args.config
    )
    SEED = args.seed
    PLOT_TRAINING = args.plot_training
    SAVE_RESULTS = args.save_results

    main(
        use_DC3_dataset,
        use_convex,
        problem_seed,
        problem_var,
        problem_nineq,
        problem_neq,
        problem_examples,
        config_path,
        SEED,
        PLOT_TRAINING,
        SAVE_RESULTS,
    )
