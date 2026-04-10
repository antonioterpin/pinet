"""Test the HardConstrainedMLP on the clipped sine function."""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import linen as nn
from flax.training import train_state

from pinet import BoxConstraint, BoxConstraintSpecification, Project, ProjectionInstance


class HardConstrainedMLP(nn.Module):
    """Simple MLP with hard constraints on the output.

    Attributes:
        box_constraint (BoxConstraint): The box constraint applied to the output.
    """

    box_constraint: BoxConstraint

    def setup(self):
        self.project = Project(box_constraint=self.box_constraint)

    @nn.compact
    def __call__(self, x, step):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = self.project.call(yraw=ProjectionInstance(x=x[..., None]))[0].x.squeeze(-1)
        return x


@pytest.mark.parametrize(
    "seed",
    [0],
)  # Add more seeds as needed
def test_clipped_sine(seed: int):
    """Test if the HardConstrainedMLP fits max(min(sin(x), 1-eps), eps).

    The training objective is to fit the sine function with a MLP, but the
    hard constraint is that the predictions must be clipped to the range
    [eps, 1 - eps]. This test checks if the projection layer effectively
    clips the predictions to the desired range.

    Args:
        seed: Random seed for reproducibility.
    """
    # Test params
    eps = 0.1
    n_samples = 1000
    learning_rate = 1e-5
    n_epochs = 10000

    # Generate dataset
    x = jnp.linspace(-jnp.pi, jnp.pi, n_samples).reshape(-1, 1)
    y = jnp.sin(x)

    # Define and initialize the hard constrained MLP
    model = HardConstrainedMLP(
        box_constraint=BoxConstraint(
            BoxConstraintSpecification(
                lb=jnp.array([eps]).reshape((1, 1, 1)),
                ub=jnp.array([1 - eps]).reshape((1, 1, 1)),
            )
        )
    )
    params = model.init(jax.random.PRNGKey(seed), jnp.ones([1, 1]), 0)
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params["params"], tx=tx
    )

    # Train the MLP
    @jax.jit
    def train_step(state, x_batch, y_batch, step):
        def loss_fn(params):
            predictions = state.apply_fn({"params": params}, x_batch, step)
            return jnp.mean((predictions - y_batch) ** 2)

        grads = jax.grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)

    for step in range(n_epochs):
        state = train_step(state, x, y, step)

    # Get predictions
    predictions = model.apply({"params": state.params}, x, 100000)

    # Clip y to meet the constraints
    clipped_y = jnp.clip(y, eps, 1 - eps)

    # Check if predictions meet the condition
    max_mean_error = 1e-1
    error = jnp.abs(predictions - clipped_y).mean()
    assert error < max_mean_error, (
        f"Predictions do not meet the clipping condition. Mean error: {error}"
    )
