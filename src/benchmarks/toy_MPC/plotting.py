"""Plotting functionalities for toy MPC."""

from collections.abc import Iterable, Sequence
from typing import cast

import cvxpy as cp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from cvxpy.constraints.constraint import Constraint as CvxpyConstraint
from flax.training.train_state import TrainState
from matplotlib.patches import Rectangle

BatchLoader = Iterable[tuple[jnp.ndarray, jnp.ndarray]]


def plot_training(
    train_loader: BatchLoader,
    valid_loader: BatchLoader,
    training_losses: Sequence[float | jnp.ndarray],
    validation_losses: Sequence[float | jnp.ndarray],
    eqcvs: Sequence[float | jnp.ndarray],
    ineqcvs: Sequence[float | jnp.ndarray],
) -> None:
    """Plot training curves.

    Args:
        train_loader: Loader for training batches.
        valid_loader: Loader for validation batches.
        training_losses: Training loss history.
        validation_losses: Validation loss history.
        eqcvs: Equality constraint violation history.
        ineqcvs: Inequality constraint violation history.
    """
    opt_train_loss = []
    for batch in train_loader:
        _, obj_batch = batch
        opt_train_loss.append(obj_batch)
    opt_train_loss = jnp.concatenate(opt_train_loss, axis=0).mean()
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1)
    plt.plot(jnp.asarray(training_losses), label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.axhline(
        y=float(opt_train_loss),
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
    plt.plot(jnp.asarray(validation_losses), label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.axhline(
        y=float(opt_valid_loss),
        color="r",
        linestyle="-",
        linewidth=2,
        label="Optimal Validation Objective",
    )
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.semilogy(jnp.asarray(eqcvs), label="Equality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Equality Violation")
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.semilogy(jnp.asarray(ineqcvs), label="Inequality Constraint Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Max Inequality Violation")
    plt.legend()

    plt.tight_layout()
    plt.show()


def generate_trajectories(
    state: TrainState,
    a: jnp.ndarray,
    lbxs: jnp.ndarray,
    ubxs: jnp.ndarray,
    lbus: jnp.ndarray,
    ubus: jnp.ndarray,
    alpha: float,
    base_dim: int,
    y_dim: int,
    dimx: int,
    xhat: jnp.ndarray,
    horizon: int,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generates trajectories from pinet and solver.

    Args:
        state: The trained model state.
        a: The equality constraint matrix.
        lbxs: Lower bounds for state variables.
        ubxs: Upper bounds for state variables.
        lbus: Lower bounds for control inputs.
        ubus: Upper bounds for control inputs.
        alpha: Regularization parameter.
        base_dim: Dimension of the base state.
        y_dim: Total dimension of the decision variable.
        dimx: Dimension of the state.
        xhat: Reference state.
        horizon: Time horizon.
        lb: Lower bounds for the state.
        ub: Upper bounds for the state.

    Returns:
        tuple: A tuple containing:
            - trajectories (jnp.ndarray): Predicted trajectories from the model.
            - trajectories_cp (jnp.ndarray): Trajectories computed using cvxpy.
    """
    ntraj = 1
    xinit = jnp.array([[-7, -5]]).reshape(ntraj, base_dim, 1)
    # Evaluate the network on these initial points
    x_init_full = jnp.concatenate(
        (xinit, jnp.zeros((xinit.shape[0], a.shape[1] - xinit.shape[1], 1))), axis=1
    )
    trajectories = state.apply_fn(
        {"params": state.params},
        xinit[:, :, 0],
        x_init_full,
        test=True,
    )
    # Solve exact problems with cvxpy
    trajectories_cp = jnp.zeros((ntraj, y_dim, 1))
    for i in range(ntraj):
        xcp = cp.Variable(y_dim)
        xinitcp = cp.Parameter(int(base_dim))
        constraints = [
            a[0] @ xcp == cp.hstack([xinitcp, np.zeros(dimx - base_dim)]),
            xcp[:dimx] >= lbxs[0, :, 0],
            xcp[:dimx] <= ubxs[0, :, 0],
            xcp[dimx:] >= lbus[0, :, 0],
            xcp[dimx:] <= ubus[0, :, 0],
        ]
        objective = cp.Minimize(
            cp.sum_squares(xcp[:dimx] - jnp.tile(xhat[:, 0], horizon + 1))
            + alpha * cp.sum_squares(xcp[dimx:])
        )
        problem = cp.Problem(objective, cast(list[CvxpyConstraint], constraints))
        # Setup problem parameter
        xinitcp.value = np.array(xinit[i, :, 0])
        problem.solve(verbose=False)
        trajectories_cp = trajectories_cp.at[i].set(jnp.asarray(xcp.value).reshape(-1, 1))

    def plot_trajectory(trajectory_pred: jnp.ndarray, trajectory_cp: jnp.ndarray) -> None:
        """Plots the trajectory in z.

        Args:
            trajectory_pred: Predicted trajectory from the model.
            trajectory_cp: Trajectory computed using cvxpy.
        """
        xpred = trajectory_pred[:dimx]
        xpred = xpred.reshape((horizon + 1, base_dim))
        # Ground truth trajectory
        xgt = trajectory_cp[:dimx]
        xgt = xgt.reshape((horizon + 1, base_dim))
        plt.plot(xpred[:, 0], xpred[:, 1], "-o", label="Prediction")
        plt.plot(xgt[:, 0], xgt[:, 1], "--*", label="Ground Truth")
        plt.plot(xhat[0], xhat[1], "rx", markersize=10, label="Goal")
        # Plot the bounds of x as a rectangle
        rect = Rectangle(
            (float(lb[0, 0, 0]), float(lb[0, 1, 0])),
            float(ub[0, 0, 0] - lb[0, 0, 0]),
            float(ub[0, 1, 0] - lb[0, 1, 0]),
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
