"""Module for setting HCNN models for the benchmarks."""

import time
from typing import Callable

import jax.numpy as jnp
from flax import linen as nn

from benchmarks.simple_QP.other_projections import (
    get_cvxpy_projection,
    get_jaxopt_projection,
)
from hcnn.constraints.affine_equality import EqualityConstraint
from hcnn.constraints.affine_inequality import AffineInequalityConstraint
from hcnn.project import Project


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
        self, x, b, sigma=1.0, omega=1.7, n_iter=100, n_iter_bwd=100, fpi=True
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
        self, x, b, sigma=1.0, omega=1.7, n_iter=100, n_iter_bwd=100, fpi=True
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


def setup_model(
    rng_key,
    hyperparameters,
    proj_method,
    A,
    X,
    G,
    h,
    setup_reps=10,
):
    """Receives problem (hyper)parameters and returns the model and its parameters."""
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
            unroll=hyperparameters["unroll"],
            equilibrate=hyperparameters["equilibrate"],
        )

        # Measure setup time
        setup_reps = 10
        start_setup_time = time.time()
        if setup_reps > 0:
            for _ in range(setup_reps):
                eq_constraint = EqualityConstraint(A=A, b=X, method=None, var_b=True)
                ineq_constraint = AffineInequalityConstraint(
                    C=G, ub=h, lb=-jnp.inf * jnp.ones_like(h)
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
        params = model.init(rng_key, x=X[:2, :, 0], b=X[:2], n_iter=2)
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
            rng_key,
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
            rng_key,
            x=X[:2, :, 0],
            b=X[:2],
        )
    else:
        raise ValueError("Projection method not valid.")

    return model, params, setup_time
