"""Module for setting HCNN models for the benchmarks."""

import time
from typing import Callable

import jax
import jax.numpy as jnp
from flax import linen as nn

from benchmarks.simple_QP.other_projections import (
    get_cvxpy_projection,
    get_jaxopt_projection,
)
from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.project import Project


def setup_pinet(A, b, C, ub, setup_reps, hyperparameters):
    """Setup of pinet projection layer."""
    eq_constraint = EqualityConstraint(A=A, b=b, method=None, var_b=True)
    ineq_constraint = AffineInequalityConstraint(
        C=C, ub=ub, lb=-jnp.inf * jnp.ones_like(ub)
    )
    projection_layer = Project(
        ineq_constraint=ineq_constraint,
        eq_constraint=eq_constraint,
        unroll=hyperparameters["unroll"],
        equilibrate=hyperparameters["equilibrate"],
    )

    # Measure setup time
    start_setup_time = time.time()
    if setup_reps > 0:
        for _ in range(setup_reps):
            eq_constraint = EqualityConstraint(A=A, b=b, method=None, var_b=True)
            ineq_constraint = AffineInequalityConstraint(
                C=C, ub=ub, lb=-jnp.inf * jnp.ones_like(ub)
            )
            _ = Project(
                ineq_constraint=ineq_constraint,
                eq_constraint=eq_constraint,
                unroll=hyperparameters["unroll"],
                equilibrate=hyperparameters["equilibrate"],
            )
        setup_time = (time.time() - start_setup_time) / setup_reps

        print(f"Time to create constraints: {setup_time:.5f} seconds")
    else:
        setup_time = 0.0

    # Define HCNN model
    if hyperparameters["unroll"]:

        def project(x, b):
            return projection_layer.call(
                y0=projection_layer.get_init(x),
                x=x,
                b=b,
                interpolation_value=0.0,
                sigma=hyperparameters["sigma"],
                omega=hyperparameters["omega"],
                n_iter=hyperparameters["n_iter_train"],
            )[0]

        def project_test(x, b):
            return projection_layer.call(
                y0=projection_layer.get_init(x),
                x=x,
                b=b,
                interpolation_value=0.0,
                sigma=hyperparameters["sigma"],
                omega=hyperparameters["omega"],
                n_iter=hyperparameters["n_iter_test"],
            )[0]

    else:

        def project(x, b):
            return projection_layer.call(
                y0=projection_layer.get_init(x),
                x=x,
                b=b,
                interpolation_value=0.0,
                sigma=hyperparameters["sigma"],
                omega=hyperparameters["omega"],
                n_iter=hyperparameters["n_iter_train"],
                n_iter_bwd=hyperparameters["n_iter_bwd"],
                fpi=hyperparameters["fpi"],
            )[0]

        def project_test(x, b):
            return projection_layer.call(
                y0=projection_layer.get_init(x),
                x=x,
                b=b,
                interpolation_value=0.0,
                sigma=hyperparameters["sigma"],
                omega=hyperparameters["omega"],
                n_iter=hyperparameters["n_iter_test"],
                n_iter_bwd=hyperparameters["n_iter_bwd"],
                fpi=hyperparameters["fpi"],
            )[0]

        return project, project_test, setup_time


def setup_jaxopt(A, b, C, ub, setup_reps, hyperparameters):
    """Setup of jaxopt projection layer."""
    # Define the jaxopt projection
    project = get_jaxopt_projection(
        A=A[0, :, :],
        C=C[0, :, :],
        d=ub[0, :, 0],
        dim=A.shape[2],
        tol=hyperparameters["jaxopt_tol"],
    )
    project_test = project
    setup_time = 0.0

    return project, project_test, setup_time


def setup_cvxpy(A, b, C, ub, setup_reps, hyperparameters):
    """Setup of cvxpy projection layer."""
    cvxpy_proj = get_cvxpy_projection(
        A=A[0, :, :],
        C=C[0, :, :],
        d=ub[0, :, 0],
        dim=A.shape[2],
    )

    def project(xx, bb):
        return cvxpy_proj(
            xx,
            bb[:, :, 0],
            solver_args={
                "verbose": False,
                "eps_abs": hyperparameters["cvxpy_tol"],
                "eps_rel": hyperparameters["cvxpy_tol"],
            },
        )[0]

    project_test = project
    setup_time = 0.0

    return project, project_test, setup_time


class HardConstrainedMLP(nn.Module):
    """Simple MLP with hard constraints on the output.

    The hard constraints are enforced through projection.
    A different projection method can be given
    for training and inference.
    """

    project: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    project_test: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    dim: int
    features_list: list
    activation: nn.Module = nn.relu

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
        if not test:
            x = self.project(x, b)
        else:
            x = self.project_test(x, b)
        return x


def setup_model(
    rng_key,
    hyperparameters,
    proj_method,
    A,
    X,
    G,
    h,
    batched_objective,
    setup_reps=10,
):
    """Receives problem (hyper)parameters and returns the model and its parameters."""
    activation = getattr(nn, hyperparameters["activation"], None)
    if activation is None:
        raise ValueError(f"Unknown activation: {hyperparameters['activation']}")

    setups = {"pinet": setup_pinet, "jaxopt": setup_jaxopt, "cvxpy": setup_cvxpy}

    if proj_method not in setups:
        raise ValueError(f"Projection method not valid: {proj_method}")

    project, project_test, setup_time = setups[proj_method](
        A=A, b=X, C=G, ub=h, setup_reps=setup_reps, hyperparameters=hyperparameters
    )

    model = HardConstrainedMLP(
        project=project,
        project_test=project_test,
        dim=A.shape[2],
        features_list=hyperparameters["features_list"],
        activation=activation,
    )
    params = model.init(
        rng_key,
        x=X[:2, :, 0],
        b=X[:2],
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

    # cvxpylayers does not support jitting
    if not proj_method == "cvxpy":
        train_step = jax.jit(train_step)

    return model, params, setup_time, train_step
