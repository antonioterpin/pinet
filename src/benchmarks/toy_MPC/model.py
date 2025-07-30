"""Module for setting up Pinet models for toy MPC."""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.box import BoxConstraint
from hcnn.project import Project
from hcnn.utils import EqualityInputs, Inputs


class PinetMLP(nn.Module):
    """Simple Pinet model with MLP backbone."""

    project: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    project_test: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    dim: int
    features_list: list
    activation: nn.Module = nn.relu
    raw: bool = False

    @nn.compact
    def __call__(
        self,
        x,
        b,
        test,
    ):
        """Call the NN."""
        for features in self.features_list:
            x = nn.Dense(features)(x)
            x = self.activation(x)
        x = nn.Dense(self.dim)(x)
        if not self.raw:
            if not test:
                x = self.project(x, b)
            else:
                x = self.project_test(x, b)
        return x


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
    projection_layer = Project(
        eq_constraint=eq_constraint,
        box_constraint=box_constraint,
        unroll=hyperparameters["unroll"],
        equilibrate=hyperparameters["equilibrate"],
    )

    # Define HCNN model
    if hyperparameters["unroll"]:

        def project(x, b):
            inp = Inputs(x=x, eq=EqualityInputs(b=b))
            return projection_layer.call(
                y0=projection_layer.get_init(inp),
                inp=inp,
                interpolation_value=0.0,
                sigma=hyperparameters["sigma"],
                omega=hyperparameters["omega"],
                n_iter=hyperparameters["n_iter_train"],
            )[0]

        def project_test(x, b):
            inp = Inputs(x=x, eq=EqualityInputs(b=b))
            return projection_layer.call(
                y0=projection_layer.get_init(inp),
                inp=inp,
                interpolation_value=0.0,
                sigma=hyperparameters["sigma"],
                omega=hyperparameters["omega"],
                n_iter=hyperparameters["n_iter_test"],
            )[0]

    else:

        def project(x, b):
            inp = Inputs(x=x, eq=EqualityInputs(b=b))
            return projection_layer.call(
                y0=projection_layer.get_init(inp),
                inp=inp,
                interpolation_value=0.0,
                sigma=hyperparameters["sigma"],
                omega=hyperparameters["omega"],
                n_iter=hyperparameters["n_iter_train"],
                n_iter_bwd=hyperparameters["n_iter_bwd"],
                fpi=hyperparameters["fpi"],
            )[0]

        def project_test(x, b):
            inp = Inputs(x=x, eq=EqualityInputs(b=b))
            return projection_layer.call(
                y0=projection_layer.get_init(inp),
                inp=inp,
                interpolation_value=0.0,
                sigma=hyperparameters["sigma"],
                omega=hyperparameters["omega"],
                n_iter=hyperparameters["n_iter_test"],
                n_iter_bwd=hyperparameters["n_iter_bwd"],
                fpi=hyperparameters["fpi"],
            )[0]

    model = PinetMLP(
        project=project,
        project_test=project_test,
        dim=A.shape[2],
        features_list=hyperparameters["features_list"],
        activation=activation,
        raw=hyperparameters["raw"],
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
