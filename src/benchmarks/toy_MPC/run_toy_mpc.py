"""Run HCNN on toy MPC problem."""

import argparse
import datetime
import pathlib
import time
import timeit
from collections.abc import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
from flax.serialization import to_bytes
from flax.training import train_state
from tqdm import tqdm

from benchmarks.toy_MPC.load_toy_mpc import load_data
from benchmarks.toy_MPC.model import setup_model
from benchmarks.toy_MPC.plotting import (
    generate_trajectories,
)
from benchmarks.toy_MPC.plotting import (
    plot_training as plot_training_curve,
)
from src.tools.utils import GracefulShutdown, Logger, load_configuration

jax.config.update("jax_enable_x64", True)

BatchLoader = Iterable[tuple[jnp.ndarray, jnp.ndarray]]


def evaluate_hcnn(
    loader: BatchLoader,
    state: train_state.TrainState,
    batched_objective: Callable[[jnp.ndarray], jnp.ndarray],
    a: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    prefix: str,
    time_evals: int = 10,
    print_res: bool = True,
    cv_tol: float = 1e-3,
    single_instance: bool = True,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    float,
    float,
    float,
]:
    """Evaluate the performance of HCNN.

    Args:
        loader: DataLoader for the dataset.
        state: The trained model state.
        batched_objective: Function to compute the objective.
        a: Coefficient matrix for equality constraints.
        lb: Lower bounds for the decision variables.
        ub: Upper bounds for the decision variables.
        prefix: Prefix for logging.
        time_evals: Number of times to evaluate the model.
        print_res: Whether to print the results.
        cv_tol: Tolerance for constraint violations.
        single_instance: Whether to evaluate a single instance.

    Returns:
        Optimal objective values, HCNN objective values, equality constraint
        violations, inequality constraint violations, percentage of valid
        constraint violations, average evaluation time, and standard deviation
        of the evaluation time.
    """
    opt_obj_batches: list[jnp.ndarray] = []
    hcnn_obj_batches: list[jnp.ndarray] = []
    eq_cv_batches: list[jnp.ndarray] = []
    ineq_cv_batches: list[jnp.ndarray] = []
    # Placeholders, overwritten by the loop; kept so the post-loop usages have
    # valid references even if the loader yielded zero batches.
    x_data = jnp.zeros((0, 0, 1))
    x_full = jnp.zeros((0, 0, 1))
    for x_data, obj in loader:
        x_full = jnp.concatenate(
            (x_data, jnp.zeros((x_data.shape[0], a.shape[1] - x_data.shape[1], 1))),
            axis=1,
        )
        predictions = state.apply_fn(
            {"params": state.params},
            x_data[:, :, 0],
            x_full,
            test=True,
        )
        opt_obj_batches.append(obj)
        hcnn_obj_batches.append(batched_objective(predictions))
        # Equality Constraint Violation
        eq_cv_batch = jnp.abs(
            a[0].reshape(1, a.shape[1], a.shape[2])
            @ predictions.reshape(x_data.shape[0], a.shape[2], 1)
            - x_full,
        )
        eq_cv_batch = jnp.max(eq_cv_batch, axis=1)
        eq_cv_batches.append(eq_cv_batch)
        # Inequality Constraint Violation
        ineq_cv_batch_ub = jnp.maximum(
            predictions.reshape(x_data.shape[0], a.shape[2], 1) - ub, 0
        )
        ineq_cv_batch_lb = jnp.maximum(
            lb - predictions.reshape(x_data.shape[0], a.shape[2], 1), 0
        )
        # Compute the maximum and normalize by the size
        ineq_cv_batch = jnp.maximum(ineq_cv_batch_ub, ineq_cv_batch_lb) / ub
        ineq_cv_batch = jnp.max(ineq_cv_batch, axis=1)
        ineq_cv_batches.append(ineq_cv_batch)
    # Objectives
    opt_obj = jnp.concatenate(opt_obj_batches, axis=0)
    opt_obj_mean = opt_obj.mean()
    hcnn_obj = jnp.concatenate(hcnn_obj_batches, axis=0)
    hcnn_obj_mean = hcnn_obj.mean()
    # Equality Constraints
    eq_cv = jnp.concatenate(eq_cv_batches, axis=0)
    eq_cv_mean = eq_cv.mean()
    eq_cv_max = eq_cv.max()
    # Inequality Constraints
    ineq_cv = jnp.concatenate(ineq_cv_batches, axis=0)
    ineq_cv_mean = ineq_cv.mean()
    ineq_cv_max = ineq_cv.max()
    ineq_perc = float((1 - jnp.mean(ineq_cv > cv_tol)) * 100)
    # Inference time (assumes all the data in one batch)
    if single_instance:
        x_inf = x_data[:1, :, :]
        x_inf_full = jnp.concatenate(
            (x_inf, jnp.zeros((x_inf.shape[0], a.shape[1] - x_inf.shape[1], 1))),
            axis=1,
        )
    else:
        x_inf = x_data
        x_inf_full = x_full
    times = timeit.repeat(
        lambda: state.apply_fn(
            {"params": state.params},
            x_inf[:, :, 0],
            x_inf_full,
            test=True,
        ).block_until_ready(),
        repeat=time_evals,
        number=1,
    )
    eval_time = float(np.mean(times))
    eval_time_std = float(np.std(times))
    if print_res:
        print(f"=========== {prefix} performance ===========")
        print("Mean objective                : ", f"{hcnn_obj_mean:.5f}")
        print(
            "Mean|Max eq. cv               : ",
            f"{eq_cv_mean:.5f}",
            "|",
            f"{eq_cv_max:.5f}",
        )
        print(
            "Mean|Max normalized ineq. cv  : ",
            f"{ineq_cv_mean:.5f}",
            "|",
            f"{ineq_cv_max:.5f}",
        )
        print(
            "Perc of valid cv. tol.        : ",
            f"{ineq_perc:.3f}%",
        )
        print("Time for evaluation [s]       : ", f"{eval_time:.5f}")
        print("Optimal mean objective        : ", f"{opt_obj_mean:.5f}")

    return (opt_obj, hcnn_obj, eq_cv, ineq_cv, ineq_perc, eval_time, eval_time_std)


def main(
    filepath: str,
    config_path: str,
    seed: int,
    plot_training: bool,
    save_results: bool,
    use_jax_loader: bool,
    run_name: str,
) -> train_state.TrainState:
    """Main for running toy MPC benchmarks.

    Args:
        filepath: Path to the dataset file.
        config_path: Path to the configuration file.
        seed: Random seed for reproducibility.
        plot_training: Whether to plot training curves.
        save_results: Whether to save the results.
        use_jax_loader: Whether to use JAX DataLoader or PyTorch DataLoader.
        run_name: Name of the run for logging.

    Returns:
        The trained model state.
    """
    hyperparameters = load_configuration(config_path)
    key = jax.random.PRNGKey(seed)
    loader_key, key = jax.random.split(key, 2)
    # Parse data
    (
        a,
        lbxs,
        ubxs,
        lbus,
        ubus,
        _xhat,
        _alpha,
        _horizon,
        _base_dim,
        x_data,
        train_loader,
        valid_loader,
        test_loader,
        batched_objective,
    ) = load_data(
        filepath=filepath,
        rng_key=loader_key,
        val_split=hyperparameters["val_split"],
        test_split=hyperparameters["test_split"],
        batch_size=hyperparameters["batch_size"],
        use_jax_loader=use_jax_loader,
    )

    y_dim = a.shape[2]
    # The X contains only the initial conditions.
    # To properly define the equality constraints we need to append zeros
    x_full = jnp.concatenate(
        (x_data, jnp.zeros((x_data.shape[0], a.shape[1] - x_data.shape[1], 1))),
        axis=1,
    )
    lb = jnp.concatenate((lbxs, lbus), axis=1)
    ub = jnp.concatenate((ubxs, ubus), axis=1)
    # Setup projection layer
    learning_rate = hyperparameters["learning_rate"]
    # Setup the model
    model, params, train_step = setup_model(
        rng_key=key,
        hyperparameters=hyperparameters,
        a=a,
        x_data=x_data,
        b=x_full,
        lb=lb,
        ub=ub,
        batched_objective=batched_objective,
    )
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params["params"], tx=tx
    )

    n_epochs = hyperparameters["n_epochs"]
    eval_every = 1
    start = time.time()
    trainig_losses: list[float | jnp.ndarray] = []
    validation_losses: list[float | jnp.ndarray] = []
    eqcvs: list[float | jnp.ndarray] = []
    ineqcvs: list[float | jnp.ndarray] = []

    with (
        Logger(run_name=run_name, project_name="hcnn_toy_mpc") as data_logger,
        GracefulShutdown("Stop detected, finishing epoch...") as g,
    ):
        data_logger.run.config.update(hyperparameters)
        for step in (pbar := tqdm(range(n_epochs))):
            if g.stop:
                break
            epoch_loss: list[jnp.ndarray] = []
            batch_sizes: list[int] = []
            start_epoch_time = time.time()
            for batch in train_loader:
                x_batch, _ = batch
                x_batch_full = jnp.concatenate(
                    (
                        x_batch,
                        jnp.zeros((x_batch.shape[0], a.shape[1] - x_batch.shape[1], 1)),
                    ),
                    axis=1,
                )
                loss, state = train_step(
                    state,
                    x_batch[:, :, 0],
                    x_batch_full,
                )
                batch_sizes.append(x_batch.shape[0])
                epoch_loss.append(loss)
            weighted_epoch_loss = sum(
                el * bs for el, bs in zip(epoch_loss, batch_sizes, strict=False)
            ) / sum(batch_sizes)
            trainig_losses.append(weighted_epoch_loss)
            pbar.set_description(f"Train Loss: {weighted_epoch_loss:.5f}")
            epoch_time = time.time() - start_epoch_time

            if step % eval_every == 0:
                start_evaluation_time = time.time()
                # TODO: Use some of the evaluate functions?
                for x_valid, valid_obj in valid_loader:
                    x_valid_full = jnp.concatenate(
                        (
                            x_valid,
                            jnp.zeros(
                                (x_valid.shape[0], a.shape[1] - x_valid.shape[1], 1)
                            ),
                        ),
                        axis=1,
                    )
                    predictions = state.apply_fn(
                        {"params": state.params},
                        x_valid[:, :, 0],
                        x_valid_full,
                        test=True,
                    )
                    validation_loss = batched_objective(predictions)
                    eqcv = jnp.abs(
                        a[0] @ predictions.reshape(-1, y_dim, 1) - x_valid_full
                    ).max()
                    ineqcvub = jnp.max(
                        jnp.maximum(predictions.reshape(-1, y_dim, 1) - ub, 0), axis=1
                    )
                    ineqcvlb = jnp.max(
                        jnp.maximum(lb - predictions.reshape(-1, y_dim, 1), 0), axis=1
                    )
                    ineqcv = jnp.maximum(ineqcvub, ineqcvlb).mean()
                    eqcvs.append(eqcv)
                    ineqcvs.append(ineqcv)
                    validation_losses.append(validation_loss.mean())
                    eval_time = time.time() - start_evaluation_time
                    pbar.set_postfix(
                        {
                            "eqcv": f"{eqcv:.5f}",
                            "ineqcv": f"{ineqcv:.5f}",
                            "Valid. Loss:": f"{validation_loss.mean():.5f}",
                        }
                    )
                    data_logger.log(
                        step,
                        {
                            "weighted_epoch_loss": weighted_epoch_loss,
                            "epoch_training_time": epoch_time,
                            "validation_objective_mean": validation_loss.mean(),
                            "validation_average_rs": (
                                (validation_loss - valid_obj) / jnp.abs(valid_obj)
                            ).mean(),
                            "validation_cv": jnp.maximum(ineqcv, eqcv),
                            "validation_time": eval_time,
                        },
                    )
        training_time = time.time() - start
        print(f"Training time: {training_time:.5f} seconds")

        if plot_training:
            plot_training_curve(
                train_loader,
                valid_loader,
                trainig_losses,
                validation_losses,
                eqcvs,
                ineqcvs,
            )
        _ = evaluate_hcnn(
            loader=valid_loader,
            state=state,
            batched_objective=batched_objective,
            prefix="Validation",
            a=a,
            lb=lb,
            ub=ub,
            cv_tol=1e-3,
            single_instance=False,
        )
        opt_obj, hcnn_obj, eq_cv, ineq_cv, ineq_perc, mean_inf_time, std_inf_time = (
            evaluate_hcnn(
                loader=test_loader,
                state=state,
                batched_objective=batched_objective,
                prefix="Test",
                a=a,
                lb=lb,
                ub=ub,
                cv_tol=1e-3,
                time_evals=10,
                single_instance=False,
            )
        )
        _, _, _, _, _, mean_inf_time_single, _std_inf_time_single = evaluate_hcnn(
            loader=test_loader,
            state=state,
            batched_objective=batched_objective,
            prefix="Test",
            a=a,
            lb=lb,
            ub=ub,
            cv_tol=1e-3,
            time_evals=10,
            single_instance=True,
        )

        # Log summary metrics for wandb
        rs = (hcnn_obj - opt_obj) / jnp.abs(opt_obj)
        cv = jnp.maximum(eq_cv, ineq_cv)
        cvthres = 1e-3
        data_logger.run.summary.update(
            {
                "Average RS Test": jnp.mean(rs),
                "Max CV Test": jnp.max(cv),
                "Percentage CV < tol": (1 - jnp.mean(cv > cvthres)) * 100,
                "Average Single Inference Time": mean_inf_time_single,
                "Average Batch Inference Time": mean_inf_time,
            }
        )

    if save_results:
        current_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = "results.npz"
        timestamp_folder = pathlib.Path(__file__).parent / "results" / current_timestamp
        timestamp_folder.mkdir(parents=True, exist_ok=True)
        results_path = timestamp_folder / results_filename
        # Save the inference time and trajectories
        jnp.savez(
            file=results_path,
            opt_obj=opt_obj,
            hcnn_obj=hcnn_obj,
            eq_cv=eq_cv,
            ineq_cv=ineq_cv,
            ineq_perc=ineq_perc,
            inference_time_mean=mean_inf_time,
            inference_time_std=std_inf_time,
            config_path=config_path,
            **hyperparameters,
        )
        # Save the network parameters for reusing
        params_filename = "params.msgpack"
        params_path = timestamp_folder / params_filename
        with open(params_path, "wb") as f:
            f.write(to_bytes(state.params))

    return state


if __name__ == "__main__":

    def parse_args():
        """Parse CLI arguments.

        Returns:
            Parsed command-line arguments.
        """
        parser = argparse.ArgumentParser(description="Run HCNN on toy MPC problem.")
        parser.add_argument(
            "--filename",
            type=str,
            required=True,
            help="Filename of dataset.",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="toy_MPC.yaml",
            help="Configuration file for HCNN hyperparameters.",
        )
        parser.add_argument(
            "--seed", type=int, default=42, help="Seed for training HCNN."
        )
        parser.add_argument(
            "--plot-training",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Plot training curves.",
        )
        parser.add_argument(
            "--save-results", action="store_true", help="Save the results."
        )
        parser.add_argument(
            "--no-save-results",
            action="store_false",
            dest="save_results",
            help="Don't save the results.",
        )
        parser.add_argument(
            "--use-saved",
            action="store_true",
            help="Use saved network to plot trajectories and print results.",
        )
        parser.add_argument(
            "--results-folder",
            type=str,
            required=False,
            default=None,
            help="Name (suffix) of the results file and params file.",
        )
        parser.add_argument(
            "--jax-loader",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use the jax loader or not. If not, use pytorch loader.",
        )
        parser.set_defaults(save_results=True)
        parser.set_defaults(use_saved=False)
        return parser.parse_args()

    # Parse arguments
    args = parse_args()
    filepath = pathlib.Path(__file__).parent.resolve() / "datasets" / args.filename
    config_path = (
        pathlib.Path(__file__).parent.parent.resolve()
        / "configs"
        / (args.config + ".yaml")
    )
    seed = args.seed
    torch.manual_seed(seed)
    use_jax_loader = args.jax_loader
    run_name = f"toy_MPC_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not args.use_saved:
        _ = main(
            filepath=filepath,
            config_path=config_path,
            seed=seed,
            plot_training=args.plot_training,
            save_results=args.save_results,
            use_jax_loader=use_jax_loader,
            run_name=run_name,
        )
    else:
        if args.results_folder is None:
            raise ValueError("Please provide the name of the results file.")

        hyperparameters = load_configuration(config_path)
        key = jax.random.PRNGKey(seed)
        loader_key, key = jax.random.split(key, 2)
        # Parse data
        (
            a,
            lbxs,
            ubxs,
            lbus,
            ubus,
            xhat,
            alpha,
            horizon,
            base_dim,
            x_data,
            train_loader,
            valid_loader,
            test_loader,
            batched_objective,
        ) = load_data(
            filepath=filepath,
            val_split=hyperparameters["val_split"],
            test_split=hyperparameters["test_split"],
            batch_size=hyperparameters["batch_size"],
            rng_key=loader_key,
            use_jax_loader=use_jax_loader,
        )
        y_dim = a.shape[2]
        # The X contains only the initial conditions.
        # To properly define the equality constraints we need to append zeros
        x_full = jnp.concatenate(
            (x_data, jnp.zeros((x_data.shape[0], a.shape[1] - x_data.shape[1], 1))),
            axis=1,
        )
        dimx = lbxs.shape[1]
        dimu = lbus.shape[1]
        lb = jnp.concatenate((lbxs, lbus), axis=1)
        ub = jnp.concatenate((ubxs, ubus), axis=1)
        model, params, train_step = setup_model(
            rng_key=key,
            hyperparameters=hyperparameters,
            a=a,
            x_data=x_data,
            b=x_full,
            lb=lb,
            ub=ub,
            batched_objective=batched_objective,
        )

        params_filepath = (
            pathlib.Path(__file__).parent.resolve()
            / "results"
            / args.results_folder
            / ("params.msgpack")
        )
        # Load saved parameters.
        with open(params_filepath, "rb") as f:
            loaded_bytes = f.read()
        from flax.serialization import (  # Import here if not already imported.
            from_bytes,
        )

        restored_params = from_bytes(params["params"], loaded_bytes)

        # Create the optimizer and state.
        tx = optax.adam(learning_rate=hyperparameters["learning_rate"])
        state = train_state.TrainState.create(
            apply_fn=model.apply, params=restored_params, tx=tx
        )

        trajectories_pred, trajectories_cp = generate_trajectories(
            state=state,
            a=a,
            lbxs=lbxs,
            ubxs=ubxs,
            lbus=lbus,
            ubus=ubus,
            alpha=alpha,
            base_dim=base_dim,
            y_dim=y_dim,
            dimx=dimx,
            xhat=xhat,
            horizon=horizon,
            lb=lb,
            ub=ub,
        )

        # Print results
        results_filepath = (
            pathlib.Path(__file__).parent.resolve()
            / "results"
            / args.results_folder
            / "results.npz"
        )
        results = jnp.load(results_filepath)
        print(
            f"Inference Time: {results['inference_time_mean']:.5f} ± "
            f"{results['inference_time_std']:.5f} s"
        )
        rel_suboptimality = (results["hcnn_obj"] - results["opt_obj"]) / results[
            "opt_obj"
        ]
        print(f"Average Relative Suboptimality: {rel_suboptimality.mean():.5%}")
        print(f"Percentage of ineq. constraint satisfaction: {results['ineq_perc']:.2f}%")

        if True:
            trajectories_path = (
                pathlib.Path(__file__).parent.resolve()
                / "results"
                / args.results_folder
                / "trajectories"
            )
            trajectories_path.mkdir(parents=True, exist_ok=True)
            for ii in range(trajectories_pred.shape[0]):
                xpred = (
                    trajectories_pred[ii, :][:dimx].reshape((horizon + 1, base_dim))
                    / 20.0
                    + 0.5
                )
                xgt = (
                    trajectories_cp[ii, :][:dimx].reshape((horizon + 1, base_dim)) / 20.0
                    + 0.5
                )
                # Save trajectory to CSV file
                # Create output directory if not exists
                # Stack the columns:
                # x (xpred[:,0]), y (xpred[:,1]), xgt (xgt[:,0]), ygt (xgt[:,1])
                data = np.column_stack((xpred[:, 0], xpred[:, 1], xgt[:, 0], xgt[:, 1]))
                csv_filename = trajectories_path / f"trajectory_{ii + 1}.csv"
                np.savetxt(
                    csv_filename,
                    data,
                    delimiter=",",
                    header="x,y,xgt,ygt",
                    comments="",
                    fmt="%.5f",
                )
