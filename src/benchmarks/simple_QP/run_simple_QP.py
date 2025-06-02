"""Run HCNN on simple QP problem."""

# TODO: Enable saving/serializing the trained network when saving results.
# TODO: Do some ablations on the ADMM parameters.
# TODO: Enable training with SoftMLP.
import argparse
import datetime
import pathlib
import time
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import torch
import yaml
from flax import linen as nn
from flax.serialization import to_bytes
from flax.training import train_state
from tqdm import tqdm

from benchmarks.simple_QP.load_simple_QP import load_data
from benchmarks.simple_QP.other_projections import (
    get_cvxpy_projection,
    get_jaxopt_projection,
)
from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.project import Project

jax.config.update("jax_enable_x64", True)


# Import hyperparameters
# Load hyperparameters from a yaml file
def load_yaml(file_path: str) -> dict:
    """Load hyperparameters for HCNN."""
    with open(file_path, "r") as file:
        hyperparameters = yaml.safe_load(file)
    return hyperparameters


def plotting(
    train_loader,
    valid_loader,
    trainig_losses,
    validation_losses,
    eqcvs,
    ineqcvs,
    eval_every,
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
        _, obj_batch = batch
        opt_valid_loss.append(obj_batch)
    opt_valid_loss = jnp.array(opt_valid_loss).mean()
    plt.subplot(1, 4, 2)
    plt.plot(
        jnp.arange(len(validation_losses), dtype=jnp.int32) * eval_every,
        validation_losses,
        label="Validation Loss",
    )
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
    plt.plot(
        jnp.arange(len(validation_losses), dtype=jnp.int32) * eval_every,
        eqcvs,
        label="Equality Constraint Violation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Max Equality Violation")

    plt.subplot(1, 4, 4)
    plt.plot(
        jnp.arange(len(validation_losses), dtype=jnp.int32) * eval_every,
        ineqcvs,
        label="Inequality Constraint Violation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Max Inequality Violation")

    plt.tight_layout()
    plt.show()


class LoggingDict:
    """Class to log results during training."""

    def __init__(self):
        """Initialize the logging dictionary."""
        self.dict = {
            "optimal_objective": [],
            "objective": [],
            "eqcv": [],
            "ineqcv": [],
            "train_time": [],
            "inf_time": [],
        }

    def update(self, optimal_objective, objective, eqcv, ineqcv, train_time, inf_time):
        """Update the logging dictionary."""
        self.dict["optimal_objective"].append(optimal_objective)
        self.dict["objective"].append(objective)
        self.dict["eqcv"].append(eqcv)
        self.dict["ineqcv"].append(ineqcv)
        self.dict["train_time"].append(train_time)
        self.dict["inf_time"].append(inf_time)

    def as_array(self, label):
        """Return the logging dictionary label as a jnp array."""
        return jnp.array(self.dict[label])


# Evaluation function to clean up code
def evaluate_hcnn(
    loader,
    state,
    sigma,
    omega,
    n_iter,
    batched_objective,
    A,
    G,
    h,
    prefix,
    time_evals=10,
    tol_cv=1e-3,
    print_res=True,
    single_instance=False,
    instances=None,
    proj_method="pinet",
):
    """Evaluate the performance of the HCNN."""
    if proj_method == "pinet":

        def predict(xx):
            return state.apply_fn(
                {"params": state.params},
                xx[:, :, 0],
                xx,
                100000,
                sigma=sigma,
                omega=omega,
                n_iter=n_iter,
            )

    else:

        def predict(xx):
            return state.apply_fn(
                {"params": state.params},
                xx[:, :, 0],
                xx,
            )

    # This assumes the loader handles all the data in one batch.
    for X, obj in loader:
        predictions = predict(X)
    opt_obj = obj.mean()
    # HCNN objective
    hcnn_obj = batched_objective(predictions)
    rs = jnp.mean((hcnn_obj - obj) / jnp.abs(obj))
    # Equality constraint violation
    eq_cv = jnp.max(
        jnp.abs(
            A[0].reshape(1, A.shape[1], A.shape[2])
            @ predictions.reshape(X.shape[0], A.shape[2], 1)
            - X
        ),
        axis=1,
    )
    # Average and max inequality constraint violation
    ineq_cv = jnp.max(
        jnp.maximum(
            G[0].reshape(1, G.shape[1], G.shape[2])
            @ predictions.reshape(X.shape[0], G.shape[2], 1)
            - h,
            0,
        ),
        axis=1,
    )
    # Percentage of constraint satisfaction at tolerance
    perc_cv = (1 - jnp.mean(ineq_cv > tol_cv)) * 100
    # Computation time
    if time_evals > 0:
        # Batch size 1 or full
        if single_instance:
            if instances is None:
                raise ValueError("Single instance evaluation requires instances.")

            eval_times = []
            for ii in instances:
                for rep in range(time_evals + 1):
                    Xtime = X[ii : ii + 1, :, :]
                    start = time.time()
                    predict(Xtime).block_until_ready()
                    # Drop first time cause it includes setups
                    if rep > 0:
                        eval_times.append(time.time() - start)
        else:
            Xtime = X
            eval_times = []
            for rep in range(time_evals + 1):
                start = time.time()
                predict(Xtime).block_until_ready()
                # Drop first time cause it includes setups
                if rep > 0:
                    eval_times.append(time.time() - start)

        eval_times = jnp.array(eval_times)
        eval_time = jnp.mean(eval_times)
    else:
        eval_time = -1
        eval_times = []
    if print_res:
        hcnn_obj_mean = hcnn_obj.mean()
        eq_cv_mean = eq_cv.mean()
        eq_cv_max = eq_cv.max()
        ineq_cv_mean = ineq_cv.mean()
        ineq_cv_max = ineq_cv.max()
        print(f"=========== {prefix} performance ===========")
        print("Mean Relative Suboptimality   : ", f"{rs:.5f}")
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
        print("Percentage of ineq. cv < tol  : ", f"{perc_cv:.5f} %")
        print("Time for evaluation [s]       : ", f"{eval_time:.5f}")
        print("Optimal mean objective        : ", f"{opt_obj:.5f}")

    return (obj, hcnn_obj, eq_cv, ineq_cv, eval_times)


# Evaluate individual instance
def evaluate_instance(
    problem_idx,
    loader,
    state,
    sigma,
    omega,
    n_iter,
    use_DC3_dataset,
    batched_objective,
    A,
    G,
    h,
    prefix,
    proj_method="pinet",
):
    """Evaluate performance on single problem instance."""
    # Evaluate HCNN solution
    # This assumes the loader handles all the data in one batch.
    for X, obj in loader:
        pass

    if proj_method == "pinet":
        predictions = state.apply_fn(
            {"params": state.params},
            X[problem_idx, :, 0].reshape((1, X.shape[1])),
            X[problem_idx].reshape((1, X.shape[1], 1)),
            100000,
            sigma=sigma,
            omega=omega,
            n_iter=n_iter,
        )
    else:
        predictions = state.apply_fn(
            {"params": state.params},
            X[problem_idx, :, 0].reshape((1, X.shape[1])),
            X[problem_idx].reshape((1, X.shape[1], 1)),
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


class HardConstrainedMLP_unroll(nn.Module):
    """Simple MLP with hard constraints on the output.

    Assumes that unrolling is used for backpropagation.
    This is defined in the projection layer.
    """

    project: Project
    features_list: list
    activation: nn.Module = nn.relu

    @nn.compact
    def __call__(
        self, x, b, step, sigma=1.0, omega=1.7, n_iter=100, n_iter_bwd=100, fpi=True
    ):
        """Call the NN."""
        for features in self.features_list:
            x = nn.Dense(features)(x)
            x = self.activation(x)
        x = nn.Dense(self.project.dim)(x)
        x = self.project.call(
            self.project.get_init(x),
            x,
            b,
            interpolation_value=0.0,
            sigma=sigma,
            omega=omega,
            n_iter=n_iter,
        )[0]
        return x


class HardConstrainedMLP_impl(nn.Module):
    """Simple MLP with hard constraints on the output.

    Assumes that implicit differentiation is used for backpropagation.
    This is defined in the projection layer.
    """

    project: Project
    features_list: list
    activation: nn.Module = nn.relu

    @nn.compact
    def __call__(
        self, x, b, step, sigma=1.0, omega=1.7, n_iter=100, n_iter_bwd=100, fpi=True
    ):
        """Call the NN."""
        for features in self.features_list:
            x = nn.Dense(features)(x)
            x = self.activation(x)
        x = nn.Dense(self.project.dim)(x)
        x = self.project.call(
            self.project.get_init(x),
            x,
            b,
            interpolation_value=0.0,
            sigma=sigma,
            omega=omega,
            n_iter=n_iter,
            n_iter_bwd=n_iter_bwd,
            fpi=fpi,
        )[0]
        return x


class HardConstrainedMLP_other(nn.Module):
    """Simple MLP with hard constraints on the output.

    Uses jaxopt or cvxpylayers for the projection.
    """

    project: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    dim: int
    features_list: list
    activation: nn.Module = nn.relu

    @nn.compact
    def __call__(
        self,
        x,
        b,
    ):
        """Call the NN."""
        for features in self.features_list:
            x = nn.Dense(features)(x)
            x = self.activation(x)
        x = nn.Dense(self.dim)(x)
        x = self.project(x, b)
        return x


def main(
    use_DC3_dataset,
    use_convex,
    problem_seed,
    problem_var,
    problem_nineq,
    problem_neq,
    problem_examples,
    use_jax_loader,
    config_path,
    SEED,
    proj_method,
    PLOT_TRAINING,
    SAVE_RESULTS,
):
    """Main for running simple QP benchmarks."""
    # Load hyperparameter configuration
    hyperparameters = load_yaml(config_path)
    unroll = hyperparameters["unroll"]
    torch.manual_seed(SEED)
    key = jax.random.PRNGKey(SEED)
    loader_key, key = jax.random.split(key, 2)
    # Load problem data
    (filename, Q, p, A, G, h, X, train_loader, valid_loader, test_loader) = load_data(
        use_DC3_dataset,
        use_convex,
        problem_seed,
        problem_var,
        problem_nineq,
        problem_neq,
        problem_examples,
        loader_key,
        use_jax_loader=use_jax_loader,
    )

    # Dimension of decision variable
    # Y_DIM = Q.shape[2]
    # Dimension of parameter vector
    LEARNING_RATE = hyperparameters["learning_rate"]
    activation = getattr(nn, hyperparameters["activation"], None)
    if activation is None:
        raise ValueError(f"Unknown activation: {hyperparameters['activation']}")

    if proj_method == "pinet":
        # Setup the projection layer
        eq_constraint = EqualityConstraint(A=A, b=X, method=None, var_b=True)
        ineq_constraint = AffineInequalityConstraint(
            C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h)
        )
        projection_layer = Project(
            ineq_constraint=ineq_constraint,
            eq_constraint=eq_constraint,
            unroll=unroll,
            equilibrate=hyperparameters["equilibrate"],
        )

        # Measure setup time
        SETUP_REPS = 10
        start_setup_time = time.time()
        if SETUP_REPS > 0:
            for _ in range(SETUP_REPS):
                eq_constraint = EqualityConstraint(A=A, b=X, method=None, var_b=True)
                ineq_constraint = AffineInequalityConstraint(
                    C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h)
                )
                _ = Project(
                    ineq_constraint=ineq_constraint,
                    eq_constraint=eq_constraint,
                    unroll=unroll,
                    equilibrate=hyperparameters["equilibrate"],
                )
            setup_time = (time.time() - start_setup_time) / SETUP_REPS

            print(f"Time to create constraints: {setup_time:.5f} seconds")
        else:
            setup_time = -1

        # Define HCNN model
        if unroll:
            model = HardConstrainedMLP_unroll(
                project=projection_layer,
                features_list=hyperparameters["features_list"],
                activation=activation,
            )
        else:
            model = HardConstrainedMLP_impl(
                project=projection_layer,
                features_list=hyperparameters["features_list"],
                activation=activation,
            )
        params = model.init(
            jax.random.PRNGKey(SEED), x=X[:2, :, 0], b=X[:2], step=0, n_iter=2
        )
    elif proj_method == "jaxopt":
        # Define the jaxopt projection
        jaxopt_projection = get_jaxopt_projection(
            A=A[0, :, :],
            C=G[0, :, :],
            d=h[0, :, 0],
            dim=A.shape[2],
            tol=hyperparameters["jaxopt_tol"],
        )
        model = HardConstrainedMLP_other(
            project=jaxopt_projection,
            dim=A.shape[2],
            features_list=hyperparameters["features_list"],
            activation=activation,
        )
        params = model.init(
            jax.random.PRNGKey(SEED),
            x=X[:2, :, 0],
            b=X[:2],
        )
    elif proj_method == "cvxpy":
        cvxpy_proj = get_cvxpy_projection(
            A=A[0, :, :],
            C=G[0, :, :],
            d=h[0, :, 0],
            dim=A.shape[2],
        )

        def cvxpy_projection(xx, bb):
            return cvxpy_proj(
                xx,
                bb[:, :, 0],
                solver_args={
                    "verbose": False,
                    "eps_abs": hyperparameters["cvxpy_tol"],
                    "eps_rel": hyperparameters["cvxpy_tol"],
                },
            )[0]

        model = HardConstrainedMLP_other(
            project=cvxpy_projection,
            dim=A.shape[2],
            features_list=hyperparameters["features_list"],
            activation=activation,
        )
        params = model.init(
            jax.random.PRNGKey(SEED),
            x=X[:2, :, 0],
            b=X[:2],
        )
    else:
        raise ValueError("Projection method not valid.")

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
    if proj_method == "pinet":

        @partial(jax.jit, static_argnames=["n_iter", "n_iter_bwd", "fpi"])
        def train_step(
            state, x_batch, b_batch, step, sigma, omega, n_iter, n_iter_bwd, fpi
        ):
            """Run a single training step."""

            def loss_fn(params):
                predictions = state.apply_fn(
                    {"params": params},
                    x_batch,
                    b_batch,
                    step,
                    sigma,
                    omega,
                    n_iter,
                    n_iter_bwd,
                    fpi,
                )
                return batched_objective(predictions).mean()

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return loss, state.apply_gradients(grads=grads)

    else:

        def train_step(
            state,
            x_batch,
            b_batch,
        ):
            """Run a single training step."""

            def loss_fn(params):
                predictions = state.apply_fn(
                    {"params": params},
                    x_batch,
                    b_batch,
                )
                return batched_objective(predictions).mean()

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            return loss, state.apply_gradients(grads=grads)

        if proj_method == "jaxopt":
            train_step = jax.jit(train_step)
        # cvxpylayers does not support jitting

    if proj_method == "pinet":
        # Measure compilation time
        for batch in train_loader:
            X_batch, _ = batch
        start_compilation_time = time.time()
        _ = train_step.lower(
            state,
            X_batch[:, :, 0],
            X_batch,
            0,
            hyperparameters["sigma"],
            hyperparameters["omega"],
            100,
            100,
            True,
        ).compile()
        # Note this also includes the time for one iteration
        compilation_time = time.time() - start_compilation_time

        print(f"Compilation time: {compilation_time:.5f} seconds")

    # Train the MLP
    N_EPOCHS = hyperparameters["n_epochs"]
    eval_every = 1
    start_training_time = time.time()
    n_iter_train = hyperparameters["n_iter_train"]
    n_iter_test = hyperparameters["n_iter_test"]
    n_iter_bwd = hyperparameters["n_iter_bwd"]
    fpi = hyperparameters["fpi"]
    trainig_losses = []
    validation_losses = []
    eqcvs = []
    ineqcvs = []
    logging_dict = LoggingDict()
    for step in (pbar := tqdm(range(N_EPOCHS))):
        epoch_loss = []
        batch_sizes = []
        start_epoch_time = time.time()
        for batch in train_loader:
            X_batch, _ = batch
            if proj_method == "pinet":
                loss, state = train_step(
                    state,
                    X_batch[:, :, 0],
                    X_batch,
                    step,
                    hyperparameters["sigma"],
                    hyperparameters["omega"],
                    n_iter_train,
                    n_iter_bwd,
                    fpi,
                )
            else:
                loss, state = train_step(
                    state,
                    X_batch[:, :, 0],
                    X_batch,
                )
            batch_sizes.append(X_batch.shape[0])
            epoch_loss.append(loss)
        weighted_epoch_loss = sum(
            el * bs for el, bs in zip(epoch_loss, batch_sizes)
        ) / sum(batch_sizes)
        trainig_losses.append(weighted_epoch_loss)
        pbar.set_description(f"Train Loss: {weighted_epoch_loss:.5f}")
        train_time = time.time() - start_epoch_time

        if step % eval_every == 0:
            start_evaluation_time = time.time()
            obj, hcnn_obj, eq_cv, ineq_cv, _ = evaluate_hcnn(
                loader=valid_loader,
                state=state,
                sigma=hyperparameters["sigma"],
                omega=hyperparameters["omega"],
                n_iter=n_iter_test,
                batched_objective=batched_objective,
                A=A,
                G=G,
                h=h,
                prefix="Validation",
                time_evals=0,
                print_res=False,
                proj_method=proj_method,
            )
            eqcvs.append(eq_cv.max())
            ineqcvs.append(ineq_cv.max())
            validation_losses.append(hcnn_obj.mean())
            pbar.set_postfix(
                {
                    "eqcv": f"{eq_cv.mean():.5f}",
                    "ineqcv": f"{ineq_cv.mean():.5f}",
                    "Valid. Loss:": f"{hcnn_obj.mean():.5f}",
                }
            )
            eval_time = time.time() - start_evaluation_time
            logging_dict.update(
                obj,
                hcnn_obj,
                eq_cv,
                ineq_cv,
                train_time,
                eval_time,
            )
    training_time = time.time() - start_training_time
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
            eval_every=eval_every,
        )

    # Evaluate validation performance
    _ = evaluate_hcnn(
        loader=valid_loader,
        state=state,
        sigma=hyperparameters["sigma"],
        omega=hyperparameters["omega"],
        n_iter=hyperparameters["n_iter_test"],
        batched_objective=batched_objective,
        prefix="Validation",
        A=A,
        G=G,
        h=h,
        proj_method=proj_method,
    )
    # Solve some validation individual problem
    problem_idx = 4
    evaluate_instance(
        problem_idx=problem_idx,
        loader=valid_loader,
        state=state,
        sigma=hyperparameters["sigma"],
        omega=hyperparameters["omega"],
        n_iter=hyperparameters["n_iter_test"],
        use_DC3_dataset=use_DC3_dataset,
        batched_objective=batched_objective,
        A=A,
        G=G,
        h=h,
        prefix="Validation",
        proj_method=proj_method,
    )

    (obj_test, obj_fun_test, eq_viol_test, ineq_viol_test, batch_inference_times) = (
        evaluate_hcnn(
            loader=test_loader,
            state=state,
            sigma=hyperparameters["sigma"],
            omega=hyperparameters["omega"],
            n_iter=hyperparameters["n_iter_test"],
            batched_objective=batched_objective,
            prefix="Testing",
            A=A,
            G=G,
            h=h,
            proj_method=proj_method,
        )
    )
    # Evaluate for single inference time
    (_, _, _, _, single_inference_times) = evaluate_hcnn(
        loader=test_loader,
        state=state,
        sigma=hyperparameters["sigma"],
        omega=hyperparameters["omega"],
        n_iter=hyperparameters["n_iter_test"],
        batched_objective=batched_objective,
        prefix="Testing",
        A=A,
        G=G,
        h=h,
        single_instance=True,
        instances=list(range(10)),
        proj_method=proj_method,
    )
    # Solve some test problems
    problem_idx = 0
    evaluate_instance(
        problem_idx=problem_idx,
        loader=test_loader,
        state=state,
        sigma=hyperparameters["sigma"],
        omega=hyperparameters["omega"],
        n_iter=hyperparameters["n_iter_test"],
        use_DC3_dataset=use_DC3_dataset,
        batched_objective=batched_objective,
        A=A,
        G=G,
        h=h,
        prefix="Testing",
        proj_method=proj_method,
    )

    # Saving of overall results
    if SAVE_RESULTS:
        # Setup results path
        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_results = "results.npz"
        results_folder = (
            pathlib.Path(__file__).parent
            / "results"
            / args.id
            / args.config
            / current_timestamp
        )
        results_folder.mkdir(parents=True, exist_ok=True)
        # Save final results
        jnp.savez(
            file=results_folder / filename_results,
            inference_time=batch_inference_times,
            single_inference_time=single_inference_times,
            setup_time=setup_time,
            compilation_time=compilation_time,
            training_time=training_time,
            eq_viol_test=eq_viol_test,
            ineq_viol_test=ineq_viol_test,
            obj_fun_test=obj_fun_test,
            opt_obj_test=obj_test,
            config_path=config_path,
            **hyperparameters,
        )
        # Save learning curve results
        jnp.savez(
            file=results_folder / "learning_curves.npz",
            optimal_objective=logging_dict.as_array("optimal_objective"),
            objective=logging_dict.as_array("objective"),
            eqcv=logging_dict.as_array("eqcv"),
            ineqcv=logging_dict.as_array("ineqcv"),
            train_time=logging_dict.as_array("train_time"),
            inf_time=logging_dict.as_array("inf_time"),
        )
        # Save network parameters
        params_filename = "params.msgpack"
        params_path = results_folder / params_filename
        with open(params_path, "wb") as f:
            f.write(to_bytes(state.params))

    return state


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
            default="benchmark_config_manual",
            help="Configuration file for hyperparameters.",
        )
        parser.add_argument(
            "--proj_method",
            type=str,
            default="pinet",
            help="Projection method. Options are: pinet, cvxpy, jaxopt.",
        )
        parser.add_argument(
            "--seed", type=int, default=42, help="Seed for training HCNN."
        )
        parser.add_argument(
            "--plot_training", action="store_true", help="Plot training curves."
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
        parser.add_argument(
            "--jax_loader",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use the jax loader or not. If not, use pytorch loader.",
        )
        parser.set_defaults(save_results=True)
        parser.set_defaults(plot_training=False)
        return parser.parse_args()

    args = parse_args()
    # Load the yaml file
    idpath = pathlib.Path(__file__).parent.resolve() / "ids" / (args.id + ".yaml")
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
    use_jax_loader = args.jax_loader
    # Configs path
    config_path = (
        pathlib.Path(__file__).parent.parent.parent.resolve()
        / "configs"
        / (args.config + ".yaml")
    )
    SEED = args.seed
    proj_method = args.proj_method
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
        use_jax_loader,
        config_path,
        SEED,
        proj_method,
        PLOT_TRAINING,
        SAVE_RESULTS,
    )
