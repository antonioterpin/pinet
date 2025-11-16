"""This module implements the solver for random second-order cone parametric problems."""

# %% Imports
import argparse
import datetime
import pathlib
import time
from typing import Optional

import jax
import numpy as np
import optax
import wandb
from flax import linen as nn
from flax.training import train_state
from jax import config as jconf
from jax import jit
from jax import numpy as jnp
from jax import random as jrnd
from jax import value_and_grad
from tqdm import tqdm

from benchmarks.QP.plotting import plot_rs_vs_cv
from benchmarks.QP.run_QP import LoggingDict
from benchmarks.toy_SOC.generate_socp import (
    constraint_violation_eq,
    constraint_violation_soc,
    generate_problem,
    objective,
    rand_sparse_mask,
    relative_suboptimality,
)
from benchmarks.toy_SOC.model import HardConstrainedMLP
from benchmarks.toy_SOC.projection_layer import build_projection_layer
from benchmarks.toy_SOC.solver import evaluate_solver
from src.tools.utils import GracefulShutdown, Logger, load_configuration

# Use 64 bit precision for numerical stability
jconf.update("jax_enable_x64", True)

# %% Setup
CONFIG = "toy_SOC"
config_path = (
    pathlib.Path(__file__).parent.parent.resolve() / "configs" / (CONFIG + ".yaml")
)
hyperparameters = load_configuration(config_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run toy second-Order cone optimization benchmark."
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    parser.add_argument(
        "--n", type=int, default=250, help="Number of variables (default: 250)"
    )
    parser.add_argument(
        "--m", type=int, default=250, help="Number of constraints (default: 250)"
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.01, help="Sparsity level (default: 0.01)"
    )
    parser.add_argument(
        "--validation-size",
        type=int,
        default=1024,
        help="Validation set size (default: 1024)",
    )
    parser.add_argument(
        "--test-size", type=int, default=1024, help="Test set size (default: 1024)"
    )
    parser.add_argument(
        "--run-tests",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run tests.",
    )
    parser.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save results.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pinet",
        help="Solution method. "
        "Options are: pinet, cvxpylayers, cvxpy, cvxpy_parametric, scs.",
    )
    parser.add_argument(
        "--measure-setup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure setup time.",
    )
    parser.add_argument(
        "--measure-compilation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Measure compilation time.",
    )
    args = parser.parse_args()
    if args.method not in ["pinet", "cvxpylayers", "cvxpy", "cvxpy_parametric", "scs"]:
        raise ValueError(f"Unknown method: {args.method}")
    return args


def print_stats(
    x: jnp.ndarray,
    s: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    xstar: jnp.ndarray,
    A: jnp.ndarray,
):
    """Print the statistics of the solution.

    Args:
        x (jnp.ndarray): Primal solution, shape (B, n, 1).
        s (jnp.ndarray): Dual solution, shape (B, m, 1).
        b (jnp.ndarray): Right-hand side of the equality constraints, shape (B, m, 1).
        c (jnp.ndarray): Coefficients for the objective function, shape (B, n, 1).
        xstar (jnp.ndarray): Optimal primal solution, shape (B, n, 1).
        A (jnp.ndarray): Constraint matrix, shape (m, n).
    """
    cv_eq = constraint_violation_eq(A, x, s, b)
    cv_soc = constraint_violation_soc(s)
    rs = relative_suboptimality(x, xstar, c)

    print("=========== Solution statistics ===========")
    # mean, std, max, min
    print(
        f"""CV (Ax = b): {jnp.mean(cv_eq):.15f} ± {jnp.std(cv_eq):.15f}
          in [{jnp.min(cv_eq):.15f}, {jnp.max(cv_eq):.15f}]"""
    )
    print(
        f"""CV (SOC): {jnp.mean(cv_soc):.15f} ± {jnp.std(cv_soc):.15f}
          in [{jnp.min(cv_soc):.15f}, {jnp.max(cv_soc):.15f}]"""
    )
    print(
        f"""RS: {jnp.mean(rs):.15f} ± {jnp.std(rs):.15f}
          in [{jnp.min(rs):.15f}, {jnp.max(rs):.15f}]"""
    )


# %% Evaluation
def evaluate_hcnn(
    batch: dict,
    state: train_state.TrainState,
    A: jnp.ndarray,
    n: int,
    prefix: str,
    time_evals: int = 10,
    tol_cv: float = 1e-3,
    print_results: bool = True,
    single_instance: bool = False,
    instances: Optional[list] = None,
) -> tuple[
    jnp.ndarray,  # Objective values
    jnp.ndarray,  # HCNN objective values
    jnp.ndarray,  # Relative suboptimalities
    jnp.ndarray,  # Equality constraint violations
    jnp.ndarray,  # SOC constraint violations
    jnp.ndarray,  # Evaluation times
]:
    """Evaluate the perfomance of the HCNN.

    Args:
        batch (dict): Input data containing "b" and "c".
        state (TrainState): Current state of the model.
        A (jnp.ndarray): Constraint matrix, shape (m, n).
        n (int): Number of variables.
        prefix (str): Prefix for printing.
        time_evals (int): Number of evaluations for timing.
        tol_cv (float): Tolerance for constraint violation.
        print_results (bool): Whether to print the results.
        single_instance (bool): Whether to evaluate a single instance.
        instances (list, optional): List of instance indices to evaluate.

    Returns:
        tuple:
            - jnp.ndarray: Optimal objective values, shape (B, 1).
            - jnp.ndarray: HCNN objective values, shape (B, 1).
            - jnp.ndarray: Relative suboptimalities, shape (B, 1).
            - jnp.ndarray: Equality constraint violations, shape (B, 1).
            - jnp.ndarray: SOC constraint violations, shape (B, 1).
            - jnp.ndarray: Evaluation times, shape (time_evals,).
    """

    def predict(batch):
        return state.apply_fn(
            state.params,
            batch["input"],
        )

    predictions = predict(batch)
    x = predictions[:, :n]
    s = predictions[:, n:]
    hcnn_obj = objective(x, batch["input"]["c"])
    opt_obj = objective(batch["xstar"], batch["input"]["c"])
    rs = relative_suboptimality(x, batch["xstar"], batch["input"]["c"])
    eq_cv = constraint_violation_eq(A, x, s, batch["input"]["b"])
    soc_cv = constraint_violation_soc(s)[..., 0]
    perc_cv = jnp.mean((eq_cv < tol_cv) & (soc_cv < tol_cv)) * 100.0
    # Computation time
    if time_evals > 0:
        # Batch size 1 or full
        if single_instance:
            if instances is None:
                raise ValueError("Single instance evaluation requires instances.")

            eval_times = []
            for ii in instances:
                batch_time = {
                    "input": {
                        "b": batch["input"]["b"][ii : ii + 1, ...],
                        "c": batch["input"]["c"][ii : ii + 1, ...],
                    }
                }
                for rep in range(time_evals + 1):
                    start = time.time()
                    predict(batch_time).block_until_ready()
                    # Drop first time cause it includes setups
                    if rep > 0:
                        eval_times.append(time.time() - start)

        else:
            eval_times = []
            for rep in range(time_evals + 1):
                start = time.time()
                predict(batch).block_until_ready()
                # Drop first time cause it includes setups
                if rep > 0:
                    eval_times.append(time.time() - start)

        eval_times = jnp.array(eval_times)
    else:
        eval_times = []

    if print_results:
        print(f"=========== {prefix} performance ===========")
        print("Mean Relative Suboptimality   : ", f"{rs.mean():.5f}")
        print("Mean objective                : ", f"{hcnn_obj.mean():.5f}")
        print(
            "Mean|Max equality violation   : ",
            f"{eq_cv.mean():.5f}",
            "|",
            f"{eq_cv.max():.5f}",
        )
        print(
            "Mean|Max SOC violation : ",
            f"{soc_cv.mean():.5f}",
            "|",
            f"{soc_cv.max():.5f}",
        )
        print("Percentage of ineq. cv < tol  : ", f"{perc_cv:.5f} %")
        if time_evals > 0:
            print("Time for evaluation [s]       : ", f"{eval_times.mean():.5f}")
        print("Optimal mean objective        : ", f"{opt_obj.mean():.5f}")

    return opt_obj, hcnn_obj, rs, eq_cv, soc_cv, eval_times


def main():
    """Main function to run the benchmark."""
    args = parse_args()

    # Set parameters from command line arguments
    SEED = args.seed
    n = args.n
    m = args.m
    sparsity = args.sparsity
    VALIDATION_SIZE = args.validation_size
    TEST_SIZE = args.test_size
    # Instances for single inference
    instances = list(range(10))
    run_tests = args.run_tests
    save_results = args.save_results
    method = args.method
    measure_setup = args.measure_setup
    measure_compilation = args.measure_compilation
    if measure_setup:
        raise NotImplementedError("Setup time measurement not implemented yet.")
    else:
        setup_time = None
    if measure_compilation:
        raise NotImplementedError("Compilation time measurement not implemented yet.")
    else:
        compilation_time = None

    nowstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"n{n}_m{m}_sparse{sparsity}_{nowstamp}"
    # Key
    key = jrnd.PRNGKey(SEED)
    ACTIVATION = getattr(nn, hyperparameters["activation"], None)
    if ACTIVATION is None:
        raise ValueError(f"Unknown activation: {hyperparameters['activation']}")
    LAYERS = hyperparameters["features_list"]

    keyA, key = jrnd.split(key)
    A = rand_sparse_mask(keyA, (m, n), sparsity=sparsity)

    # %% Validate the problem
    B = 1024
    # Symbolic problem
    key_problem, key = jrnd.split(key)
    b, c, xstar, sstar = generate_problem(key_problem, A, B)

    # %% CVXPY
    A_np = np.asarray(A)
    if run_tests:
        x_cvxpy, s_cvxpy, _ = evaluate_solver(
            A_np=A_np,
            b_batch=b,
            c_batch=c,
            n=n,
            m=m,
            eps=1e-9,
            verbose=False,
            method="cvxpy_parametric",
        )

        # Print the statistics of the solution
        print_stats(x_cvxpy, s_cvxpy, b, c, xstar, A)

    # %% Use our solver
    Aaug = jnp.concatenate((A, jnp.eye(m)), axis=1)
    assert Aaug.shape == (
        m,
        m + n,
    ), f"Augmented matrix A should have shape ({m}, {m + n}), instead: {Aaug.shape}"

    project = build_projection_layer(
        A=Aaug, n=n, m=m, hyperparameters=hyperparameters, method=method
    )

    # %% Test the projection
    # To test the correctness of the projection, we can sample random points,
    # project them, and check if the result has no constraint violation.

    if run_tests:

        def test_projection(
            b: jnp.ndarray, c: jnp.ndarray, xstar: jnp.ndarray, sstar: jnp.ndarray
        ):
            """Test the projection function on random samples.

            Args:
                b (jnp.ndarray): RHS of the equality constraints, shape (B, m, 1).
                c (jnp.ndarray): Coefficients for the objective function, shape (B, n, 1).
                xstar (jnp.ndarray): Optimal primal solution, shape (B, n, 1).
                sstar (jnp.ndarray): Optimal dual solution, shape (B, m, 1).
            """
            n_samples = b.shape[0]
            yraw = jrnd.uniform(key, (n_samples, n + m, 1))

            x = yraw[:, :n]
            s = yraw[:, n:]
            cv_eq_raw = constraint_violation_eq(A, x, s, b)
            cv_soc_raw = constraint_violation_soc(s)
            if jnp.all(cv_eq_raw < 1e-6) and jnp.all(cv_soc_raw < 1e-6):
                print("No constraint violation in the raw samples.")

            cv_eq_opt = constraint_violation_eq(A, xstar, sstar, b)
            cv_soc_opt = constraint_violation_soc(sstar)
            if jnp.any(cv_eq_opt > 1e-6) or jnp.any(cv_soc_opt > 1e-6):
                print(f"Optimal sample: {cv_eq_opt.max()=}, {cv_soc_opt.max()=}")

            # Project the point
            y, _ = project(jnp.zeros_like(yraw), yraw, b)
            x = y[:, :n]
            s = y[:, n:]

            # Check the constraint violation
            cv_eq = constraint_violation_eq(A, x, s, b)
            cv_eq_soc = constraint_violation_soc(s)
            if (
                jnp.any(cv_eq > 1e-6)
                or jnp.any(cv_eq_soc > 1e-6)
                or jnp.isnan(cv_eq).any()
                or jnp.isnan(cv_eq_soc).any()
            ):
                print(f"Projection failed: {cv_eq.max()=}, {cv_eq_soc.max()=}")
                print_stats(x, s, b, c, xstar.reshape(-1, n, 1), A)
            print("All projections passed.")

        test_projection(b, c, xstar, sstar)

    # %% Train the MLP
    BATCH_SIZE = hyperparameters["batch_size"]
    N_EPOCHS = hyperparameters["n_epochs"]
    LEARNING_RATE = hyperparameters["learning_rate"]
    key_train, key_init = jrnd.split(key)

    # Batcher
    def make_batch(key: jax.random.PRNGKey, batch_size: int = BATCH_SIZE):
        """Generate a batch of random problems.

        Args:
            key (jax.random.PRNGKey): Random key for generating the batch.
            batch_size (int): Number of problem instances in the batch.

        Returns:
            tuple:
                - dict: Input data containing "b" and "c".
                - jnp.ndarray: Optimal primal solution, shape (B, n, 1).
                - jnp.ndarray: Optimal dual solution, shape (B, m, 1).
        """
        key_prob, key = jrnd.split(key)
        b, c, xstar, sstar = generate_problem(key_prob, A, batch_size)
        return {
            "input": {"b": b, "c": c},
            "xstar": xstar,
            "sstar": sstar,
        }, key

    # %% Initialize the model
    model = HardConstrainedMLP(
        activation=ACTIVATION, layers=LAYERS, project=project, m=m, n=n
    )

    # Sample one batch only to create shapes for initialisation
    batch, key = make_batch(key_init, batch_size=1)
    key, key_init = jrnd.split(key)
    params = model.init(key_init, batch["input"])

    tx = optax.adam(LEARNING_RATE)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # %% Generate validation and test data
    validation_batch, key_test = make_batch(key=key, batch_size=VALIDATION_SIZE)
    test_batch, _ = make_batch(key_test, batch_size=TEST_SIZE)
    # %% Benchmark solver
    if method in ["cvxpy", "cvxpy_parametric", "scs"]:
        reps = 20
        # Batch inference times
        batch_times = []
        for _ in tqdm(range(reps)):
            _, _, times = evaluate_solver(
                A_np=A_np,
                b_batch=test_batch["input"]["b"],
                c_batch=test_batch["input"]["c"],
                n=n,
                m=m,
                eps=1e-4,
                verbose=False,
                method=method,
                use_tqdm=False,
            )
            batch_times.append(jnp.sum(times))
        # Single inference times
        single_times = []
        for i in tqdm(range(reps)):
            for ii in instances:
                _, _, times = evaluate_solver(
                    A_np=A_np,
                    b_batch=test_batch["input"]["b"][ii : ii + 1, ...],
                    c_batch=test_batch["input"]["c"][ii : ii + 1, ...],
                    n=n,
                    m=m,
                    eps=1e-4,
                    verbose=False,
                    method=method,
                    use_tqdm=False,
                )
                single_times.append(times.item())
        if save_results:
            filename_results = "results.npz"
            results_folder = (
                pathlib.Path(__file__).parent
                / "results"
                / f"n{n}_m{m}"
                / method
                / nowstamp
            )
            results_folder.mkdir(parents=True, exist_ok=True)
            jnp.savez(
                file=results_folder / filename_results,
                inference_time=batch_times,
                single_inference_time=single_times,
            )

        return

    # %% Training
    def loss_fn(params: dict, input: dict):
        """Compute the loss function and auxiliary values.

        Args:
            params (dict): Model parameters.
            input (dict): Input data containing "b" and "c".

        Returns:
            tuple:
                - loss (jnp.ndarray): Mean objective value.
                - aux (tuple): Auxiliary values containing constraint violations.
        """
        c = input["c"]
        pred = model.apply(params, input)
        x = pred[:, :n]
        s = pred[:, n:]
        objective_value = objective(x, c)
        return jnp.mean(objective_value), (x, s)

    def train_step(state: train_state.TrainState, batch: dict):
        """Perform a single training step.

        Args:
            state (TrainState): Current state of the model.
            batch (dict): Input data containing "b" and "c".

        Returns:
            tuple:
                - state (TrainState): Updated state of the model.
                - loss (jnp.ndarray): Loss value after the step.
                - aux (tuple): Auxiliary values containing constraint violations.
        """
        grad_fn = value_and_grad(loss_fn, has_aux=True)
        (loss, aux), grads = grad_fn(state.params, batch["input"])
        state = state.apply_gradients(grads=grads)
        return state, loss, aux

    # Only pinet supports jitting
    if method == "pinet":
        loss_fn = jit(loss_fn)
        train_step = jit(train_step)

    # Training loop
    eval_every = 1
    logging_dict = LoggingDict()
    start_training_time = time.time()
    with (
        Logger(run_name=run_name, project_name="toy_SOC") as data_logger,
        GracefulShutdown("Stop detected, finish epoch...") as g,
    ):
        data_logger.run.config.update(hyperparameters)
        for epoch in (pbar := tqdm(range(1, N_EPOCHS + 1))):
            if g.stop:
                break
            key_train, key = jrnd.split(key_train)
            batch, key_train = make_batch(key)
            start_epoch_time = time.time()
            state, l, (x, s) = train_step(state, batch)
            pbar.set_description(f"Train Loss: {l:.5f}")
            train_time = time.time() - start_epoch_time
            if epoch % eval_every == 0 or epoch == 1:
                start_evaluation_time = time.time()
                obj, hcnn_obj, rs, cv_eq, cv_soc, _ = evaluate_hcnn(
                    batch=validation_batch,
                    state=state,
                    A=A,
                    n=n,
                    prefix="Validation",
                    time_evals=-1,
                    tol_cv=1e-3,
                    print_results=False,
                    single_instance=False,
                    instances=None,
                )
                eval_time = time.time() - start_evaluation_time
                logging_dict.update(
                    optimal_objective=obj,
                    objective=hcnn_obj,
                    eqcv=cv_eq,
                    ineqcv=cv_soc,
                    train_time=train_time,
                    inf_time=eval_time,
                )
                data_logger.log(
                    epoch,
                    {
                        "loss": l,
                        "epoch_training_time": train_time,
                        "validation_avg_rs": rs.mean(),
                        "validation_max_rs": rs.max(),
                        "validation_eq_cv": cv_eq.max(),
                        "validation_cv_soc": cv_soc.max(),
                        "validation_time": eval_time,
                    },
                )
                pbar.set_postfix(
                    {
                        "CV Eq": f"{cv_eq.max():.4e}",
                        "CV SOC": f"{cv_soc.max():.4e}",
                        "RS": f"{rs.mean():.4e}",
                    }
                )
        training_time = time.time() - start_training_time
        # %% Test
        pred_val = model.apply(state.params, test_batch["input"])
        b = test_batch["input"]["b"]

        x_pred = pred_val[:, :n]
        s_pred = pred_val[:, n:]

        print_stats(
            x_pred,
            s_pred,
            test_batch["input"]["b"],
            test_batch["input"]["c"],
            test_batch["xstar"],
            A,
        )
        # %% Batch statistics
        (
            opt_obj_test,
            hcnn_obj_test,
            rs_test,
            cv_eq_test,
            cv_soc_test,
            batch_times_tests,
        ) = evaluate_hcnn(
            batch=test_batch,
            state=state,
            A=A,
            n=n,
            prefix="Testing",
            time_evals=10,
            tol_cv=1e-3,
            print_results=True,
            single_instance=False,
            instances=None,
        )
        # %% Single Statistics
        _, _, _, _, _, single_times_tests = evaluate_hcnn(
            batch=test_batch,
            state=state,
            A=A,
            n=n,
            prefix="Testing Single Instances",
            time_evals=10,
            tol_cv=1e-3,
            print_results=True,
            single_instance=True,
            instances=instances,
        )

        # %%
        cvthres = 1e-3
        rsthres = 5e-2
        fig, rs, cv = plot_rs_vs_cv(
            obj_fun_test=hcnn_obj_test,
            obj_test=opt_obj_test,
            eq_viol_test=jnp.max(cv_eq_test, axis=1),
            ineq_viol_test=jnp.max(cv_soc_test, axis=1),
            cvthres=cvthres,
            rsthres=rsthres,
        )
        data_logger.run.log({"RS vs CV": wandb.Image(fig)})
        data_logger.run.summary.update(
            {
                "Average RS Test": jnp.mean(rs_test),
                "Max CV Test": jnp.max(cv),
                "Percentage CV < Tol": jnp.mean(cv < cvthres) * 100.0,
                "Average Single Inference Time": jnp.mean(single_times_tests),
                "Average Batch Inference Time": jnp.mean(batch_times_tests),
            }
        )

        if save_results:
            filename_results = "results.npz"
            results_folder = (
                pathlib.Path(__file__).parent
                / "results"
                / f"n{n}_m{m}"
                / method
                / nowstamp
            )
            results_folder.mkdir(parents=True, exist_ok=True)
            jnp.savez(
                file=results_folder / filename_results,
                inference_time=batch_times_tests,
                single_inference_time=single_times_tests,
                setup_time=setup_time,
                compilation_time=compilation_time,
                training_time=training_time,
                eq_viol_test=cv_eq_test,
                ineq_viol_test=cv_soc_test,
                obj_fun_test=hcnn_obj_test,
                opt_obj_test=opt_obj_test,
                config_path=config_path,
                **hyperparameters,
            )
            # Save learning curve results
            jnp.savez(
                file=results_folder / "learning_curves.npz",
                optimal_objective=logging_dict.as_array("optimal_objective"),
                objective=logging_dict.as_array("objective"),
                eqcv=logging_dict.as_array("eqcv"),
                ineqcv=logging_dict.as_array("ineqcv"),
                train_time=logging_dict.as_array("train_time"),
                inf_time=logging_dict.as_array("inf_time"),
            )
            # Save learning curve results
            jnp.savez(
                file=results_folder / "learning_curves.npz",
                optimal_objective=logging_dict.as_array("optimal_objective"),
                objective=logging_dict.as_array("objective"),
                eqcv=logging_dict.as_array("eqcv"),
                ineqcv=logging_dict.as_array("ineqcv"),
                train_time=logging_dict.as_array("train_time"),
                inf_time=logging_dict.as_array("inf_time"),
            )


if __name__ == "__main__":
    main()
