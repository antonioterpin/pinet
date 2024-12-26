"""Test the HardConstrainedMLP on the clipped sine function."""

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

from hcnn.flax_project import Project


class HardConstrainedMLP(nn.Module):
    """Simple MLP with hard constraints on the output."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.softplus(x)
        x = nn.Dense(64)(x)
        x = nn.softplus(x)
        x = nn.Dense(1)(x)
        x = Project()(x)
        return x


def test_clipped_sine():
    """Test if the HardConstrainedMLP fits max(min(sin(x), 1-EPS), EPS).

    The training objective is to fit the sine function with a MLP, but the
    hard constraint is that the predictions must be clipped to the range
    [EPS, 1 - EPS]. This test checks if the projection layer effectively
    clips the predictions to the desired range.
    """
    # Test params
    EPS = 0.1
    N_SAMPLES = 1000
    LEARNING_RATE = 1e-2
    N_EPOCHS = 10000
    SEED = 0

    # Generate dataset
    x = jnp.linspace(-jnp.pi, jnp.pi, N_SAMPLES).reshape(-1, 1)
    y = jnp.sin(x)

    # Define and initialize the hard constrained MLP
    model = HardConstrainedMLP()
    params = model.init(jax.random.PRNGKey(SEED), jnp.ones([1, 1]))
    tx = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params["params"], tx=tx
    )

    # Train the MLP
    @jax.jit
    def train_step(state, x_batch, y_batch):
        def loss_fn(params):
            predictions = state.apply_fn({"params": params}, x_batch)
            return jnp.mean((predictions - y_batch) ** 2)

        grads = jax.grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)

    for _ in range(N_EPOCHS):
        state = train_step(state, x, y)

    # Get predictions
    predictions = model.apply({"params": state.params}, x)

    # Clip y to meet the constraints
    clipped_y = jnp.clip(y, EPS, 1 - EPS)

    # Check that clipping the predictions correctly passes the test
    clipped_predictions = jnp.clip(predictions, EPS, 1 - EPS)
    assert jnp.allclose(
        clipped_predictions, clipped_y, atol=1e-2
    ), "The clipped MLP predictions do not meet the clipping condition."

    # Check if predictions meet the condition
    assert jnp.allclose(
        predictions, clipped_y, atol=1e-2
    ), "The MLP predictions do not meet the clipping condition."
