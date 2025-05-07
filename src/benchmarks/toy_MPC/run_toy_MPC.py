"""Run HCNN on toy MPC problem."""

import argparse
import datetime
import os
import pathlib
import time
import timeit
from functools import partial

import cvxpy as cp
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import yaml
from flax import linen as nn
from flax.serialization import to_bytes
from flax.training import train_state
from tqdm import tqdm

from benchmarks.toy_MPC.load_toy_MPC import ToyMPCDataset, create_dataloaders
from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.box import BoxConstraint
from hcnn.project import Project

# TODO: Run experiments with new configurations.
jax.config.update("jax_enable_x64", True)


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
        _, obj_batch = batch
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
    plt.semilogy(eqcvs, label="Equality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Equality Violation")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.semilogy(ineqcvs, label="Inequality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Inequality Violation")
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_hcnn(
    loader,
    state,
    sigma,
    omega,
    n_iter,
    batched_objective,
    A,
    lb,
    ub,
    prefix,
    time_evals=10,
    print_res=True,
    cv_tol=1e-3,
    single_instance=True,
):
    """Evaluate the performance of the HCNN."""
    opt_obj = []
    hcnn_obj = []
    eq_cv = []
    ineq_cv = []
    for X, obj in loader:
        X_full = jnp.concatenate(
            (X, jnp.zeros((X.shape[0], A.shape[1] - X.shape[1], 1))), axis=1
        )
        predictions = state.apply_fn(
            {"params": state.params},
            X[:, :, 0],
            X_full,
            100000,
            sigma=sigma,
            omega=omega,
            n_iter=n_iter,
        )
        opt_obj.append(obj)
        hcnn_obj.append(batched_objective(predictions))
        # Equality Constraint Violation
        eq_cv_batch = jnp.abs(
            A[0].reshape(1, A.shape[1], A.shape[2])
            @ predictions.reshape(X.shape[0], A.shape[2], 1)
            - X_full,
        )
        eq_cv_batch = jnp.max(eq_cv_batch, axis=1)
        eq_cv.append(eq_cv_batch)
        # Inequality Constraint Violation
        ineq_cv_batch_ub = jnp.maximum(
            predictions.reshape(X.shape[0], A.shape[2], 1) - ub, 0
        )
        ineq_cv_batch_lb = jnp.maximum(
            lb - predictions.reshape(X.shape[0], A.shape[2], 1), 0
        )
        # Compute the maximum and normalize by the size
        ineq_cv_batch = jnp.maximum(ineq_cv_batch_ub, ineq_cv_batch_lb) / ub
        ineq_cv_batch = jnp.max(ineq_cv_batch, axis=1)
        ineq_cv.append(ineq_cv_batch)
    # Objectives
    opt_obj = jnp.concatenate(opt_obj, axis=0)
    opt_obj_mean = opt_obj.mean()
    hcnn_obj_mean = jnp.concatenate(hcnn_obj, axis=0).mean()
    # Equality Constraints
    eq_cv = jnp.concatenate(eq_cv, axis=0)
    eq_cv_mean = eq_cv.mean()
    eq_cv_max = eq_cv.max()
    # Inequality Constraints
    ineq_cv = jnp.concatenate(ineq_cv, axis=0)
    ineq_cv_mean = ineq_cv.mean()
    ineq_cv_max = ineq_cv.max()
    ineq_perc = (1 - jnp.mean(ineq_cv > cv_tol)) * 100
    # Inference time (assumes all the data in one batch)
    if single_instance:
        X_inf = X[:1, :, :]
        X_inf_full = jnp.concatenate(
            (X_inf, jnp.zeros((X_inf.shape[0], A.shape[1] - X_inf.shape[1], 1))), axis=1
        )
    else:
        X_inf = X
        X_inf_full = X_full
    times = timeit.repeat(
        lambda: state.apply_fn(
            {"params": state.params},
            X_inf[:, :, 0],
            X_inf_full,
            100000,
            sigma=sigma,
            omega=omega,
            n_iter=n_iter,
        ).block_until_ready(),
        repeat=time_evals,
        number=1,
    )
    eval_time = np.mean(times)
    eval_time_std = np.std(times)
    if print_res:
        print(f"=========== {prefix} performance ===========")
        print("Mean objective                : ", f"{hcnn_obj_mean:.5f}")
        print(
            "Mean|Max eq. cv               : ",
            f"{eq_cv_mean:.5f}",
            "|",
            f"{eq_cv_max:.5f}",
        )
        print(
            "Mean|Max normalized ineq. cv  : ",
            f"{ineq_cv_mean:.5f}",
            "|",
            f"{ineq_cv_max:.5f}",
        )
        print(
            "Perc of valid cv. tol.        : ",
            f"{ineq_perc:.3f}%",
        )
        print("Time for evaluation [s]       : ", f"{eval_time:.5f}")
        print("Optimal mean objective        : ", f"{opt_obj_mean:.5f}")

    return (opt_obj, hcnn_obj, eq_cv, ineq_cv, ineq_perc, eval_time, eval_time_std)


def evaluate_instance(
    problem_idx,
    loader,
    state,
    sigma,
    omega,
    n_iter,
    batched_objective,
    A,
    lb,
    ub,
    prefix,
):
    """Evaluate performance on single problem instance."""
    X = loader.dataset.dataset.x0sets[
        loader.dataset.indices[problem_idx : problem_idx + 1]
    ]
    X_full = jnp.concatenate(
        (X, jnp.zeros((X.shape[0], A.shape[1] - X.shape[1], 1))), axis=1
    )
    predictions = state.apply_fn(
        {"params": state.params},
        X[:, :, 0],
        X_full,
        100000,
        sigma=sigma,
        omega=omega,
        n_iter=n_iter,
    )

    objective_val_hcnn = batched_objective(predictions).item()
    eqcv_val_hcnn = jnp.abs(
        A[0].reshape(1, A.shape[1], A.shape[2])
        @ predictions.reshape(X.shape[0], A.shape[2], 1)
        - X_full,
    ).max()
    ineqcv_ub_val_hcnn = jnp.maximum(predictions.reshape(1, -1, 1) - ub, 0).max()
    ineqcv_lb_val_hcnn = jnp.maximum(lb - predictions.reshape(1, -1, 1), 0).max()
    ineqcv_val_hcnn = jnp.maximum(ineqcv_ub_val_hcnn, ineqcv_lb_val_hcnn)
    print(f"=========== {prefix} individual performance ===========")
    print("HCNN")
    print(f"Objective:  \t{objective_val_hcnn:.5e}")
    print(f"Eq. cv:     \t{eqcv_val_hcnn:.5e}")
    print(f"Ineq. cv:   \t{ineqcv_val_hcnn:.5e}")

    objective_val = loader.dataset.dataset.objectives[
        loader.dataset.indices[problem_idx]
    ]
    eqcv_val = jnp.abs(
        A[0].reshape(1, A.shape[1], A.shape[2])
        @ loader.dataset.dataset.Ystar[loader.dataset.indices[problem_idx]].reshape(
            X.shape[0], A.shape[2], 1
        )
        - X_full
    ).max()
    ineqcv_ub_val = jnp.maximum(
        loader.dataset.dataset.Ystar[loader.dataset.indices[problem_idx]].reshape(
            1, -1, 1
        )
        - ub,
        0,
    ).max()
    ineqcv_lb_val = jnp.maximum(
        lb
        - loader.dataset.dataset.Ystar[loader.dataset.indices[problem_idx]].reshape(
            1, -1, 1
        ),
        0,
    ).max()
    ineqcv_val = jnp.maximum(ineqcv_ub_val, ineqcv_lb_val)

    print("Optimal Solution")
    print(f"Objective:  \t{objective_val:.5e}")
    print(f"Eq. cv:     \t{eqcv_val:.5e}")
    print(f"Ineq. cv:   \t{ineqcv_val:.5e}")


def load_data(filepath):
    """Load problem data."""
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", filepath)
    ToyDataset = ToyMPCDataset(dataset_path)
    train_loader, valid_loader, test_loader = create_dataloaders(
        dataset_path, batch_size=2000, val_split=0.1, test_split=0.1
    )
    As, lbxs, ubxs, lbus, ubus, xhat, alpha, T, base_dim = ToyDataset.const
    X = ToyDataset.x0sets

    return (
        As,
        lbxs,
        ubxs,
        lbus,
        ubus,
        xhat,
        alpha,
        T,
        base_dim,
        X,
        train_loader,
        valid_loader,
        test_loader,
    )


def generate_trajectories(
    state, sigma, omega, n_iter_test, As, lbxs, ubxs, lbus, ubus, alpha
):
    """Generates trajectories from HCNN and solver."""
    ntraj = 1
    xinit = jnp.array([[-7, -5]]).reshape(ntraj, base_dim, 1)
    # Evaluate the network on these initial points
    Xinitfull = jnp.concatenate(
        (xinit, jnp.zeros((xinit.shape[0], As.shape[1] - xinit.shape[1], 1))), axis=1
    )
    trajectories = state.apply_fn(
        {"params": state.params},
        xinit[:, :, 0],
        Xinitfull,
        100000,
        sigma=sigma,
        omega=omega,
        n_iter=n_iter_test,
    )
    # Solve exact problems with cvxpy
    trajectories_cp = jnp.zeros((ntraj, Y_DIM, 1))
    for i in range(ntraj):
        xcp = cp.Variable(Y_DIM)
        xinitcp = cp.Parameter(base_dim.item())
        constraints = [
            As[0] @ xcp == cp.hstack([xinitcp, np.zeros(dimx - base_dim)]),
            xcp[:dimx] >= lbxs[0, :, 0],
            xcp[:dimx] <= ubxs[0, :, 0],
            xcp[dimx:] >= lbus[0, :, 0],
            xcp[dimx:] <= ubus[0, :, 0],
        ]
        objective = cp.Minimize(
            cp.sum_squares(xcp[:dimx] - jnp.tile(xhat[:, 0], T + 1))
            + alpha * cp.sum_squares(xcp[dimx:])
        )
        problem = cp.Problem(objective, constraints)
        # Setup problem parameter
        xinitcp.value = np.array(xinit[i, :, 0])
        problem.solve(verbose=True)
        trajectories_cp = trajectories_cp.at[i].set(
            jnp.asarray(xcp.value).reshape(-1, 1)
        )

    def plot_trajectory(trajectory_pred, trajectory_cp):
        """Plots the trajectory in z."""
        xpred = trajectory_pred[:dimx]
        xpred = xpred.reshape((T + 1, base_dim))
        # Ground truth trajectory
        xgt = trajectory_cp[:dimx]
        xgt = xgt.reshape((T + 1, base_dim))
        plt.plot(xpred[:, 0], xpred[:, 1], "-o", label="Prediction")
        plt.plot(xgt[:, 0], xgt[:, 1], "--*", label="Ground Truth")
        plt.plot(xhat[0], xhat[1], "rx", markersize=10, label="Goal")
        # Plot the bounds of x as a rectangle
        rect = plt.Rectangle(
            (lb[0, 0, 0], lb[0, 1, 0]),
            ub[0, 0, 0] - lb[0, 0, 0],
            ub[1, 0, 0] - lb[1, 0, 0],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            linestyle="--",
            label="Bounds",
        )
        plt.gca().add_patch(rect)
        plt.legend()
        plt.show()

    for ii in range(ntraj):
        plot_trajectory(trajectories[ii, :], trajectories_cp[ii, :, 0])

    return trajectories, trajectories_cp


class HardConstrainedMLP_unroll(nn.Module):
    """Simple MLP with hard constraints on the output.

    Assumes that unrolling is used for backpropagation.
    This is defined in the projection layer.
    """

    project: Project

    def setup(self):
        """Setup for each NN call."""
        self.schedule = optax.linear_schedule(0.0, 0.0, 70, 0)

    @nn.compact
    def __call__(
        self,
        x,
        b,
        step,
        sigma=1.0,
        omega=1.7,
        n_iter=100,
        n_iter_bwd=100,
        fpi=True,
        raw=False,
    ):
        """Call the NN."""
        x = nn.Dense(200)(x)
        x = nn.relu(x)
        x = nn.Dense(200)(x)
        x = nn.relu(x)
        x = nn.Dense(self.project.dim)(x)
        alpha = self.schedule(step)
        if not raw:
            x = self.project.call(
                self.project.get_init(x),
                x,
                b,
                interpolation_value=alpha,
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

    def setup(self):
        """Setup for each NN call."""
        self.schedule = optax.linear_schedule(0.0, 0.0, 70, 0)

    @nn.compact
    def __call__(
        self,
        x,
        b,
        step,
        sigma=1.0,
        omega=1.7,
        n_iter=100,
        n_iter_bwd=100,
        fpi=True,
        raw=False,
    ):
        """Call the NN."""
        x = nn.Dense(200)(x)
        x = nn.relu(x)
        x = nn.Dense(200)(x)
        x = nn.relu(x)
        x = nn.Dense(self.project.dim)(x)
        alpha = self.schedule(step)
        if not raw:
            x = self.project.call(
                self.project.get_init(x),
                x,
                b,
                interpolation_value=alpha,
                sigma=sigma,
                omega=omega,
                n_iter=n_iter,
                n_iter_bwd=n_iter_bwd,
                fpi=fpi,
            )[0]
        return x


def main(
    args,
    filepath,
    config_path,
    SEED,
    PLOT_TRAINING,
    SAVE_RESULTS,
):
    """Main for running toy MPC benchmarks."""
    hyperparameters = load_yaml(config_path)
    unroll = hyperparameters["unroll"]
    # Parse data
    (
        As,
        lbxs,
        ubxs,
        lbus,
        ubus,
        xhat,
        alpha,
        T,
        base_dim,
        X,
        train_loader,
        valid_loader,
        test_loader,
    ) = load_data(filepath)
    Y_DIM = As.shape[2]
    # The X contains only the initial conditions.
    # To properly define the equality constraints we need to append zeros
    Xfull = jnp.concatenate(
        (X, jnp.zeros((X.shape[0], As.shape[1] - X.shape[1], 1))), axis=1
    )
    dimx = lbxs.shape[1]
    lb = jnp.concatenate((lbxs, lbus), axis=1)
    ub = jnp.concatenate((ubxs, ubus), axis=1)
    # Setup projection layer
    LEARNING_RATE = hyperparameters["learning_rate"]
    eq_constraint = EqualityConstraint(A=As, b=Xfull, method="pinv", var_b=True)
    box_constraint = BoxConstraint(
        lower_bound=lb,
        upper_bound=ub,
    )
    # ineq_constraint = AffineInequalityConstraint(C=C, lb=lb, ub=ub)
    projection_layer = Project(
        box_constraint=box_constraint,
        eq_constraint=eq_constraint,
        unroll=unroll,
    )
    if unroll:
        model = HardConstrainedMLP_unroll(project=projection_layer)
    else:
        model = HardConstrainedMLP_impl(project=projection_layer)
    params = model.init(
        jax.random.PRNGKey(SEED), x=X[:2, :, 0], b=Xfull[:2], step=0, n_iter=2
    )
    tx = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params["params"], tx=tx
    )

    def quadratic_form(prediction):
        """Evaluate the quadratic objective."""
        return jnp.sum(
            (prediction[:dimx] - jnp.tile(xhat[:, 0], T + 1)) ** 2
        ) + alpha * jnp.sum(prediction[dimx:] ** 2)

    batched_objective = jax.vmap(quadratic_form, in_axes=[0])

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

    N_EPOCHS = hyperparameters["n_epochs"]
    start = time.time()
    n_iter_train = hyperparameters["n_iter_train"]
    n_iter_test = hyperparameters["n_iter_test"]
    n_iter_bwd = hyperparameters["n_iter_bwd"]
    fpi = hyperparameters["fpi"]
    trainig_losses = []
    validation_losses = []
    eqcvs = []
    ineqcvs = []

    for step in (pbar := tqdm(range(N_EPOCHS))):
        epoch_loss = []
        for batch in train_loader:
            X_batch, _ = batch
            X_batch_full = jnp.concatenate(
                (
                    X_batch,
                    jnp.zeros((X_batch.shape[0], As.shape[1] - X_batch.shape[1], 1)),
                ),
                axis=1,
            )
            loss, state = train_step(
                state,
                X_batch[:, :, 0],
                X_batch_full,
                step,
                hyperparameters["sigma"],
                hyperparameters["omega"],
                n_iter_train,
                n_iter_bwd,
                fpi,
            )
            epoch_loss.append(loss)
        pbar.set_description(f"Train Loss: {jnp.array(epoch_loss).mean():.5f}")
        trainig_losses.append(jnp.array(epoch_loss).mean())

        if step % 1 == 0:
            for X_valid, _ in valid_loader:
                X_valid_full = jnp.concatenate(
                    (
                        X_valid,
                        jnp.zeros(
                            (X_valid.shape[0], As.shape[1] - X_valid.shape[1], 1)
                        ),
                    ),
                    axis=1,
                )
                predictions = state.apply_fn(
                    {"params": state.params},
                    X_valid[:, :, 0],
                    X_valid_full,
                    step,
                    hyperparameters["sigma"],
                    hyperparameters["omega"],
                    n_iter=n_iter_test,
                )
                loss = batched_objective(predictions).mean()
                eqcv = jnp.abs(
                    As[0] @ predictions.reshape(-1, Y_DIM, 1) - X_valid_full
                ).max()
                ineqcvub = jnp.max(
                    jnp.maximum(predictions.reshape(-1, Y_DIM, 1) - ub, 0), axis=1
                )
                ineqcvlb = jnp.max(
                    jnp.maximum(lb - predictions.reshape(-1, Y_DIM, 1), 0), axis=1
                )
                ineqcv = jnp.maximum(ineqcvub, ineqcvlb).mean()
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

    def plot_instance_trajectory(loader, problem_idx, raw=False):
        """Plots the trajectory in z."""
        X = loader.dataset.dataset.x0sets[loader.dataset.indices[problem_idx]].reshape(
            1, -1
        )
        X_full = jnp.concatenate((X, jnp.zeros((1, As.shape[1] - X.shape[1]))), axis=1)
        predictions = state.apply_fn(
            {"params": state.params},
            X,
            X_full.reshape(1, -1, 1),
            10000,
            hyperparameters["sigma"],
            hyperparameters["omega"],
            n_iter=n_iter_test,
            raw=raw,
        )
        # Predicted trajectory
        xpred = predictions[problem_idx, :][:dimx]
        xpred = xpred.reshape((T + 1, base_dim))
        # Ground truth trajectory
        xgt = loader.dataset.dataset.Ystar[loader.dataset.indices[problem_idx]][:dimx]
        xgt = xgt.reshape((T + 1, base_dim))
        plt.plot(xpred[:, 0], xpred[:, 1], "-o", label="Prediction")
        plt.plot(xgt[:, 0], xgt[:, 1], "--*", label="Ground Truth")
        plt.plot(xhat[0], xhat[1], "rx", markersize=10, label="Goal")
        # Plot the bounds of x as a rectangle
        rect = plt.Rectangle(
            (lb[0, 0, 0], lb[0, 1, 0]),
            ub[0, 0, 0] - lb[0, 0, 0],
            ub[1, 0, 0] - lb[1, 0, 0],
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            linestyle="--",
            label="Bounds",
        )
        plt.gca().add_patch(rect)
        plt.legend()
        plt.show()

    problem_idx = 0
    plot_instance_trajectory(test_loader, problem_idx)
    if PLOT_TRAINING:
        plotting(
            train_loader,
            valid_loader,
            trainig_losses,
            validation_losses,
            eqcvs,
            ineqcvs,
        )
    _ = evaluate_hcnn(
        loader=valid_loader,
        state=state,
        sigma=hyperparameters["sigma"],
        omega=hyperparameters["omega"],
        n_iter=n_iter_test,
        batched_objective=batched_objective,
        prefix="Validation",
        A=As,
        lb=lb,
        ub=ub,
        cv_tol=1e-3,
        single_instance=False,
    )
    problem_idx = 18
    evaluate_instance(
        problem_idx=problem_idx,
        loader=valid_loader,
        state=state,
        sigma=hyperparameters["sigma"],
        omega=hyperparameters["omega"],
        n_iter=200,
        batched_objective=batched_objective,
        A=As,
        lb=lb,
        ub=ub,
        prefix="Validation",
    )
    opt_obj, hcnn_obj, eq_cv, ineq_cv, ineq_perc, mean_inf_time, std_inf_time = (
        evaluate_hcnn(
            loader=test_loader,
            state=state,
            sigma=hyperparameters["sigma"],
            omega=hyperparameters["omega"],
            n_iter=n_iter_test,
            batched_objective=batched_objective,
            prefix="Test",
            A=As,
            lb=lb,
            ub=ub,
            cv_tol=1e-3,
            time_evals=10,
            single_instance=True,
        )
    )
    problem_idx = 0
    evaluate_instance(
        problem_idx=problem_idx,
        loader=test_loader,
        state=state,
        sigma=hyperparameters["sigma"],
        omega=hyperparameters["omega"],
        n_iter=hyperparameters["n_iter_test"],
        batched_objective=batched_objective,
        A=As,
        lb=lb,
        ub=ub,
        prefix="Testing",
    )

    if SAVE_RESULTS:
        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = "results.npz"
        timestamp_folder = pathlib.Path(__file__).parent / "results" / current_timestamp
        timestamp_folder.mkdir(parents=True, exist_ok=True)
        results_path = timestamp_folder / results_filename
        # Save the inference time and trajectories
        jnp.savez(
            file=results_path,
            opt_obj=opt_obj,
            hcnn_obj=hcnn_obj,
            eq_cv=eq_cv,
            ineq_cv=ineq_cv,
            ineq_perc=ineq_perc,
            inference_time_mean=mean_inf_time,
            inference_time_std=std_inf_time,
            config_path=config_path,
            **hyperparameters,
        )
        # Save the network parameters for reusing
        params_filename = "params.msgpack"
        params_path = timestamp_folder / params_filename
        with open(params_path, "wb") as f:
            f.write(to_bytes(state.params))

    return state


if __name__ == "__main__":

    def parse_args():
        """Parse CLI arguments."""
        parser = argparse.ArgumentParser(description="Run HCNN on toy MPC problem.")
        parser.add_argument(
            "--filename",
            type=str,
            required=True,
            help="Filename of dataset.",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="toy_MPC.yaml",
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
        parser.add_argument(
            "--use_saved",
            action="store_true",
            help="Use saved network to plot trajectories and print results.",
        )
        parser.add_argument(
            "--results_folder",
            type=str,
            required=False,
            default=None,
            help="Name (suffix) of the results file and params file.",
        )
        parser.set_defaults(save_results=True)
        parser.set_defaults(use_saved=False)
        return parser.parse_args()

    # Parse arguments
    args = parse_args()
    filepath = pathlib.Path(__file__).parent.resolve() / "datasets" / args.filename
    config_path = (
        pathlib.Path(__file__).parent.parent.parent.resolve() / "configs" / args.config
    )
    SEED = args.seed
    if not args.use_saved:
        _ = main(
            args,
            filepath,
            config_path,
            SEED,
            args.plot_training,
            args.save_results,
        )
    else:
        if args.results_folder is None:
            raise ValueError("Please provide the name of the results file.")

        hyperparameters = load_yaml(config_path)
        unroll = hyperparameters["unroll"]
        # Parse data
        (
            As,
            lbxs,
            ubxs,
            lbus,
            ubus,
            xhat,
            alpha,
            T,
            base_dim,
            X,
            train_loader,
            valid_loader,
            test_loader,
        ) = load_data(filepath)
        Y_DIM = As.shape[2]
        # The X contains only the initial conditions.
        # To properly define the equality constraints we need to append zeros
        Xfull = jnp.concatenate(
            (X, jnp.zeros((X.shape[0], As.shape[1] - X.shape[1], 1))), axis=1
        )
        dimx = lbxs.shape[1]
        dimu = lbus.shape[1]
        lb = jnp.concatenate((lbxs, lbus), axis=1)
        ub = jnp.concatenate((ubxs, ubus), axis=1)
        eq_constraint = EqualityConstraint(A=As, b=Xfull, method="pinv", var_b=True)
        box_constraint = BoxConstraint(
            lower_bound=lb,
            upper_bound=ub,
        )
        projection_layer = Project(
            box_constraint=box_constraint,
            eq_constraint=eq_constraint,
            unroll=unroll,
        )
        if unroll:
            model = HardConstrainedMLP_unroll(project=projection_layer)
        else:
            model = HardConstrainedMLP_impl(project=projection_layer)
        # Initialize the model to create a parameter structure.
        params = model.init(
            jax.random.PRNGKey(SEED), x=X[:2, :, 0], b=Xfull[:2], step=0, n_iter=2
        )

        params_filepath = (
            pathlib.Path(__file__).parent.resolve()
            / "results"
            / args.results_folder
            / ("params.msgpack")
        )
        # Load saved parameters.
        with open(params_filepath, "rb") as f:
            loaded_bytes = f.read()
        from flax.serialization import (  # Import here if not already imported.
            from_bytes,
        )

        restored_params = from_bytes(params["params"], loaded_bytes)

        # Create the optimizer and state.
        tx = optax.adam(learning_rate=hyperparameters["learning_rate"])
        state = train_state.TrainState.create(
            apply_fn=model.apply, params=restored_params, tx=tx
        )

        trajectories_pred, trajectories_cp = generate_trajectories(
            state,
            hyperparameters["sigma"],
            hyperparameters["omega"],
            hyperparameters["n_iter_test"],
            As,
            lbxs,
            ubxs,
            lbus,
            ubus,
            alpha,
        )

        # Print results
        results_filepath = (
            pathlib.Path(__file__).parent.resolve()
            / "results"
            / args.results_folder
            / "results.npz"
        )
        results = jnp.load(results_filepath)
        print(
            f"Inference Time: {results['inference_time_mean']:.5f} ± "
            f"{results['inference_time_std']:.5f} s"
        )
        rel_suboptimality = (results["hcnn_obj"][0, :] - results["opt_obj"]) / results[
            "opt_obj"
        ]
        print(f"Average Relative Suboptimality: {rel_suboptimality.mean():.5%}")
        print(
            f"Percentage of ineq. constraint satisfaction: {results['ineq_perc']:.2f}%"
        )

        if True:
            trajectories_path = (
                pathlib.Path(__file__).parent.resolve()
                / "results"
                / args.results_folder
                / "trajectories"
            )
            trajectories_path.mkdir(parents=True, exist_ok=True)
            for ii in range(trajectories_pred.shape[0]):
                xpred = (
                    trajectories_pred[ii, :][:dimx].reshape((T + 1, base_dim)) / 20.0
                    + 0.5
                )
                xgt = (
                    trajectories_cp[ii, :][:dimx].reshape((T + 1, base_dim)) / 20.0
                    + 0.5
                )
                # Save trajectory to CSV file
                # Create output directory if not exists
                # Stack the columns:
                # x (xpred[:,0]), y (xpred[:,1]), xgt (xgt[:,0]), ygt (xgt[:,1])
                data = np.column_stack((xpred[:, 0], xpred[:, 1], xgt[:, 0], xgt[:, 1]))
                csv_filename = trajectories_path / f"trajectory_{ii+1}.csv"
                np.savetxt(
                    csv_filename,
                    data,
                    delimiter=",",
                    header="x,y,xgt,ygt",
                    comments="",
                    fmt="%.5f",
                )
