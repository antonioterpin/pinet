"""Module for setting HCNN models for the benchmarks."""

import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState

from pinet import (
    AffineInequalityConstraint,
    BoxConstraint,
    EqualityConstraint,
    EqualityConstraintsSpecification,
    EquilibrationParams,
    Project,
    ProjectionInstance,
)

from .other_projections import get_cvxpy_projection, get_jaxopt_projection

ActivationFn = Callable[[jax.Array], jax.Array]
ProjectionFn = Callable[[jax.Array, jax.Array], jax.Array]
TrainStepFn = Callable[[TrainState, jax.Array, jax.Array], tuple[jax.Array, TrainState]]


def setup_pinet(
    hyperparameters: dict[str, Any],
    eq_constraint: EqualityConstraint | None = None,
    ineq_constraint: AffineInequalityConstraint | None = None,
    box_constraint: BoxConstraint | None = None,
    setup_reps: int = -1,
) -> tuple[ProjectionFn, ProjectionFn, float]:
    """Setup of pinet projection layer.

    Args:
        hyperparameters: Hyperparameters for the model and projection.
        eq_constraint: Equality constraint for the projection layer.
        ineq_constraint: Affine inequality constraint for the projection layer.
        box_constraint: Box constraint for the projection layer.
        setup_reps: Number of repetitions for setup timing. If <= 0, skip timing.

    Returns:
        project (Callable): Function to project the input during training.
        project_test (Callable): Function to project the input during testing.
        setup_time (float): Time taken to set up the projection layer.
    """
    projection_layer = Project(
        ineq_constraint=ineq_constraint,
        eq_constraint=eq_constraint,
        box_constraint=box_constraint,
        unroll=hyperparameters["unroll"],
        equilibration_params=EquilibrationParams(**hyperparameters["equilibrate"]),
    )

    setup_time = 0.0
    if setup_reps > 0:
        start_setup_time = time.time()
        for _ in range(setup_reps):
            _ = Project(
                ineq_constraint=ineq_constraint,
                eq_constraint=eq_constraint,
                box_constraint=box_constraint,
                unroll=hyperparameters["unroll"],
                equilibration_params=EquilibrationParams(
                    **hyperparameters["equilibrate"]
                ),
            )
        setup_time = (time.time() - start_setup_time) / (max(setup_reps, 1))
        print(f"Time to create constraints: {setup_time:.5f} seconds")

    kw = (
        {}
        if hyperparameters["unroll"]
        else {
            "n_iter_bwd": hyperparameters["n_iter_bwd"],
            "fpi": hyperparameters["fpi"],
        }
    )

    def project(x, b):
        inp = ProjectionInstance(x=x[..., None], eq=EqualityConstraintsSpecification(b=b))
        return projection_layer.call(
            y_raw=inp,
            sigma=hyperparameters["sigma"],
            omega=hyperparameters["omega"],
            n_iter=hyperparameters["n_iter_train"],
            **kw,
        )[0].x[..., 0]

    def project_test(x, b):
        inp = ProjectionInstance(x=x[..., None], eq=EqualityConstraintsSpecification(b=b))
        return projection_layer.call(
            y_raw=inp,
            sigma=hyperparameters["sigma"],
            omega=hyperparameters["omega"],
            n_iter=hyperparameters["n_iter_test"],
            **kw,
        )[0].x[..., 0]

    return project, project_test, setup_time


def setup_jaxopt(
    a_mat: jax.Array,
    b: jax.Array,
    c_mat: jax.Array,
    ub: jax.Array,
    setup_reps: int,
    hyperparameters: dict[str, Any],
) -> tuple[ProjectionFn, ProjectionFn, float]:
    """Setup of jaxopt projection layer.

    Args:
        a_mat: Coefficient matrix for the equality constraint.
        b: Right-hand side vector for the equality constraint.
        c_mat: Coefficient matrix for the inequality constraint.
        ub: Upper bounds for the inequality constraint.
        setup_reps: Number of repetitions for setup timing.
            For jaxopt, we do not time the setup.
        hyperparameters: Hyperparameters for the projection.

    Returns:
        Function to project the input, function to project the input in test mode,
        and the setup time, which is always `0.0` for jaxopt.
    """
    project = get_jaxopt_projection(
        a_mat=a_mat[0, :, :],
        c_mat=c_mat[0, :, :],
        d=ub[0, :, 0],
        dim=a_mat.shape[2],
        tol=hyperparameters["jaxopt_tol"],
    )
    project_test = project
    setup_time = 0.0
    return project, project_test, setup_time


def setup_cvxpy(
    a_mat: jax.Array,
    b: jax.Array,
    c_mat: jax.Array,
    ub: jax.Array,
    setup_reps: int,
    hyperparameters: dict[str, Any],
) -> tuple[
    ProjectionFn,
    ProjectionFn,
    float,
]:
    """Setup of cvxpy projection layer.

    Args:
        a_mat: Coefficient matrix for the equality constraint.
        b: Right-hand side vector for the equality constraint.
        c_mat: Coefficient matrix for the inequality constraint.
        ub: Upper bounds for the inequality constraint.
        setup_reps: Number of repetitions for setup timing.
            For cvxpy, we do not time the setup.
        hyperparameters: Hyperparameters for the projection.

    Returns:
        Function to project the input, function to project the input in test mode,
        and the setup time, which is always `0.0` for cvxpy.
    """
    cvxpy_proj = get_cvxpy_projection(
        a_mat=a_mat[0, :, :],
        c_mat=c_mat[0, :, :],
        d=ub[0, :, 0],
        dim=a_mat.shape[2],
    )

    def project(xx, bb):
        return cvxpy_proj(xx, bb[:, :, 0])[0]

    project_test = project
    setup_time = 0.0
    return project, project_test, setup_time


class HardConstrainedMLP(nn.Module):
    """Simple MLP with hard constraints on the output.

    Attributes:
        project: Function to project the input during training.
        project_test: Function to project the input during testing.
        dim: Dimension of the output.
        features_list: List of features for each hidden layer.
        activation: Activation function to use in the MLP.
        raw_train: Whether to skip projection during training.
        raw_test: Whether to skip projection during testing.
    """

    project: ProjectionFn
    project_test: ProjectionFn
    dim: int
    features_list: list[int]
    activation: ActivationFn = nn.relu
    raw_train: bool = False
    raw_test: bool = False

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        b: jax.Array,
        test: bool,
    ):
        """Forward pass of the MLP with projection.

        Args:
            x: Input data.
            b: Right-hand side vector for the equality constraint.
            test: Whether to use the test projection.

        Returns:
            jax.Array: Output of the MLP after projection.
        """
        for features in self.features_list:
            x = nn.Dense(features)(x)
            x = self.activation(x)
        x = nn.Dense(self.dim)(x)
        if test and (not self.raw_test):
            x = self.project_test(x, b)
        elif (not test) and (not self.raw_train):
            x = self.project(x, b)
        return x


def build_model_and_train_step(
    *,
    rng_key: jax.Array,
    dim: int,
    features_list: list[int],
    activation: ActivationFn,
    project: ProjectionFn,
    project_test: ProjectionFn,
    raw_train: bool,
    raw_test: bool,
    loss_fn: Callable[[jax.Array, jax.Array], jax.Array],
    example_x: jax.Array,
    example_b: jax.Array,
    jit: bool = True,
) -> tuple[
    nn.Module,
    Any,
    TrainStepFn,
]:
    """Build the model and the training step function.

    Args:
        rng_key: Random key for initialization.
        dim: Dimension of the input.
        features_list: List of features for the MLP.
        activation: Activation function to use in the MLP.
        project: Function to project the input during training.
        project_test: Function to project the input during testing.
        raw_train: Whether to use raw training data without projection.
        raw_test: Whether to use raw test data without projection.
        loss_fn: Loss function to be used during training.
        example_x: Example input for model initialization.
        example_b:
            Example right-hand side vector for the equality constraint.
        jit: Whether to JIT compile the training step.

    Returns:
        The constructed model, its initial parameters, and a function to perform
        a training step.
    """
    model = HardConstrainedMLP(
        project=project,
        project_test=project_test,
        dim=dim,
        features_list=features_list,
        activation=activation,
        raw_train=raw_train,
        raw_test=raw_test,
    )

    params = model.init(
        rng_key,
        x=example_x,
        b=example_b,
        test=False,
    )

    def train_step(state, x_batch: jax.Array, b_batch: jax.Array):
        def _loss(p):
            preds = state.apply_fn({"params": p}, x=x_batch, b=b_batch, test=False)
            return loss_fn(preds, b_batch).mean()

        loss, grads = jax.value_and_grad(_loss)(state.params)
        return loss, state.apply_gradients(grads=grads)

    if jit:
        train_step = jax.jit(train_step)

    return model, params, train_step


def setup_model(
    rng_key: jax.Array,
    hyperparameters: dict[str, Any],
    proj_method: str,
    a_mat: jax.Array,
    x_data: jax.Array,
    g_mat: jax.Array,
    h: jax.Array,
    batched_loss: Callable[[jax.Array, jax.Array], jax.Array],
    setup_reps: int = 10,
) -> tuple[
    nn.Module,
    Any,
    float,
    TrainStepFn,
]:
    """Receives problem (hyper)parameters and returns the model and its parameters.

    Args:
        rng_key: Random key for initialization.
        hyperparameters: Hyperparameters for the model and projection.
        proj_method: Method for projection (`"pinet"`, `"jaxopt"`, or `"cvxpy"`).
        a_mat: Coefficient matrix for the equality constraint.
        x_data: Right-hand side vector for the equality constraint.
        g_mat: Coefficient matrix for the inequality constraint.
        h: Upper bounds for the inequality constraint.
        batched_loss: Loss function to be used during training.
        setup_reps: Number of repetitions for setup timing.

    Returns:
        The constructed model, its initial parameters, the setup time for the
        projection layer, and a function to perform a training step.

    Raises:
        ValueError: If the activation name or projection method is invalid.
    """
    activation = getattr(nn, hyperparameters["activation"], None)
    if activation is None:
        raise ValueError(f"Unknown activation: {hyperparameters['activation']}")

    if proj_method == "pinet":
        eq_constraint = EqualityConstraint(a_mat=a_mat, b=x_data, method=None, var_b=True)
        ineq_constraint = AffineInequalityConstraint(
            c_mat=g_mat, ub=h, lb=-jnp.inf * jnp.ones_like(h)
        )
        project, project_test, setup_time = setup_pinet(
            eq_constraint=eq_constraint,
            ineq_constraint=ineq_constraint,
            hyperparameters=hyperparameters,
        )
    elif proj_method == "jaxopt":
        project, project_test, setup_time = setup_jaxopt(
            a_mat=a_mat,
            b=x_data,
            c_mat=g_mat,
            ub=h,
            setup_reps=setup_reps,
            hyperparameters=hyperparameters,
        )
    elif proj_method == "cvxpy":
        project, project_test, setup_time = setup_cvxpy(
            a_mat=a_mat,
            b=x_data,
            c_mat=g_mat,
            ub=h,
            setup_reps=setup_reps,
            hyperparameters=hyperparameters,
        )
    else:
        raise ValueError(f"Projection method not valid: {proj_method}")

    model, params, train_step = build_model_and_train_step(
        rng_key=rng_key,
        dim=a_mat.shape[2],
        features_list=hyperparameters["features_list"],
        activation=activation,
        project=project,
        project_test=project_test,
        raw_train=hyperparameters.get("raw_train", False),
        raw_test=hyperparameters.get("raw_test", False),
        loss_fn=batched_loss,
        example_x=x_data[:2, :, 0],
        example_b=x_data[:2],
        jit=(proj_method != "cvxpy"),
    )

    return model, params, setup_time, train_step
