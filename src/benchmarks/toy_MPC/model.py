"""Module for setting up Pinet models for toy MPC."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from pinet import BoxConstraint, BoxConstraintSpecification, EqualityConstraint
from src.benchmarks.model import build_model_and_train_step, setup_pinet


def setup_model(
    rng_key: jax.Array,
    hyperparameters: dict[str, Any],
    a_dyn: jnp.ndarray,
    x_data: jnp.ndarray,
    b: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    batched_objective: Callable[[jnp.ndarray], jnp.ndarray],
) -> tuple[
    nn.Module,
    dict[str, Any],
    Callable[[TrainState, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, TrainState]],
]:
    """Receives problem (hyper)parameters and returns the model and its parameters.

    Args:
        rng_key: Random key for initialization.
        hyperparameters: Hyperparameters for the model.
        a_dyn: Coefficient matrix for the equality constraint.
        x_data: Input data for the model.
        b: Right-hand side vector for the equality constraint.
        lb: Lower bounds for the box constraint.
        ub: Upper bounds for the box constraint.
        batched_objective: Function to compute the objective value for the model
            predictions.

    Returns:
        model (nn.Module): The Pinet model.
        params (dict[str, Any]): Parameters of the model.
        train_step (Callable): Function to perform a training step.

    Raises:
        ValueError: If the activation function is unknown.
    """
    activation = getattr(nn, hyperparameters["activation"], None)
    if activation is None:
        raise ValueError(f"Unknown activation: {hyperparameters['activation']}")

    # Constraints + projection layer
    eq_constraint = EqualityConstraint(a_dyn=a_dyn, b=b, method=None, var_b=True)
    box_constraint = BoxConstraint(BoxConstraintSpecification(lb=lb, ub=ub))
    project, project_test, _ = setup_pinet(
        eq_constraint=eq_constraint,
        box_constraint=box_constraint,
        hyperparameters=hyperparameters,
    )

    # Reuse the shared builder; adapt the loss to ignore b
    model, params, train_step = build_model_and_train_step(
        rng_key=rng_key,
        dim=a_dyn.shape[2],
        features_list=hyperparameters["features_list"],
        activation=activation,
        project=project,
        project_test=project_test,
        raw_train=hyperparameters.get("raw_train", False),
        raw_test=hyperparameters.get("raw_test", False),
        loss_fn=lambda preds, _b: batched_objective(preds),
        example_x=x_data[:1, :, 0],
        example_b=b[:1],
        jit=True,
    )

    return model, params, train_step
