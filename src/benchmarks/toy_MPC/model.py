"""Module for setting up Pinet models for toy MPC."""

import jax
from flax import linen as nn

from pinet import BoxConstraint, EqualityConstraint
from src.benchmarks.model import HardConstrainedMLP, setup_pinet


def setup_model(
    rng_key,
    hyperparameters,
    A,
    X,
    b,
    lb,
    ub,
    batched_objective,
):
    """Receives problem (hyper)parameters and returns the model and its parameters."""
    activation = getattr(nn, hyperparameters["activation"], None)
    if activation is None:
        raise ValueError(f"Unknown activation: {hyperparameters['activation']}")

    # Setup Pinet projection layer
    eq_constraint = EqualityConstraint(A=A, b=b, method=None, var_b=True)
    box_constraint = BoxConstraint(lower_bound=lb, upper_bound=ub)
    project, project_test, _ = setup_pinet(
        eq_constraint=eq_constraint,
        box_constraint=box_constraint,
        hyperparameters=hyperparameters,
    )

    model = HardConstrainedMLP(
        project=project,
        project_test=project_test,
        dim=A.shape[2],
        features_list=hyperparameters["features_list"],
        activation=activation,
        raw_train=hyperparameters.get("raw_train", False),
        raw_test=hyperparameters.get("raw_test", False),
    )
    params = model.init(
        rng_key,
        x=X[:1, :, 0],
        b=b[:1],
        test=False,
    )

    # Setup the MLP training routine
    def train_step(
        state,
        x_batch,
        b_batch,
    ):
        """Run a single training step."""

        def loss_fn(params):
            predictions = state.apply_fn(
                {"params": params},
                x=x_batch,
                b=b_batch,
                test=False,
            )
            return batched_objective(predictions).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return loss, state.apply_gradients(grads=grads)

    return model, params, train_step
