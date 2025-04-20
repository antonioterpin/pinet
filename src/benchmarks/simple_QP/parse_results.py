"""Simple script to parse results and present in table."""

import pathlib

import jax.numpy as jnp
import pandas as pd
from tabulate import tabulate

# TODO: Improve this script to automatically find the available files.
# TODO: Maybe enabling writing the output in a nice Latex
#   formatted file.
# Choose data
# Convex high accuracy
problem = "nonconvex_low"
if problem == "convex_high":
    label = "Convex Objective - High Accuracy"
    filename_bicgstab = (
        "results_20250312_181354_"
        "dc3_random_simple_dataset_"
        "var100_ineq50_eq50_ex10000.npz"
    )
    filename_fpi = (
        "results_20250312_181513_"
        "dc3_random_simple_dataset_"
        "var100_ineq50_eq50_ex10000.npz"
    )
    filename_unroll = (
        "results_20250312_181700_"
        "dc3_random_simple_dataset_"
        "var100_ineq50_eq50_ex10000.npz"
    )
    filename_dc3 = "dc3_simple.npz"
    path = pathlib.Path(__file__).parent.resolve() / "results" / "convex_high"
    solver_time = 3.915

# Convex low accuracy
elif problem == "convex_low":
    label = "Convex Objective - Low Accuracy"
    filename_bicgstab = (
        "results_20250313_123433_dc3_random_simple_dataset_"
        "var100_ineq50_eq50_ex10000.npz"
    )
    filename_fpi = (
        "results_20250313_123532_dc3_random_simple_dataset_"
        "var100_ineq50_eq50_ex10000.npz"
    )
    filename_unroll = (
        "results_20250313_123749_dc3_random_simple_dataset_"
        "var100_ineq50_eq50_ex10000.npz"
    )
    filename_dc3 = "dc3_simple.npz"
    path = pathlib.Path(__file__).parent.resolve() / "results" / "convex_low"
    solver_time = 3.915

# Nonconvex high
elif problem == "nonconvex_high":
    label = "NonConvex Objective - High Accuracy"
    filename_bicgstab = (
        "results_20250312_181818_dc3_random_nonconvex_dataset_"
        "var100_ineq50_eq50_ex10000.npz"
    )
    filename_fpi = (
        "results_20250312_181937_dc3_random_nonconvex_dataset_"
        "var100_ineq50_eq50_ex10000.npz"
    )
    filename_unroll = (
        "results_20250312_182124_dc3_random_nonconvex_dataset_"
        "var100_ineq50_eq50_ex10000.npz"
    )
    filename_dc3 = "dc3_nonconvex.npz"
    path = pathlib.Path(__file__).parent.resolve() / "results" / "nonconvex_high"
    solver_time = 483.14

# Nonconvex low
elif problem == "nonconvex_low":
    label = "NonConvex Objective - Low Accuracy"
    filename_bicgstab = (
        "results_20250313_123915_dc3_random_nonconvex_dataset_"
        "var100_ineq50_eq50_ex10000.npz"
    )
    filename_fpi = (
        "results_20250313_124012_dc3_random_nonconvex_dataset_"
        "var100_ineq50_eq50_ex10000.npz"
    )
    filename_unroll = (
        "results_20250313_124228_dc3_random_nonconvex_dataset_"
        "var100_ineq50_eq50_ex10000.npz"
    )
    filename_dc3 = "dc3_nonconvex.npz"
    path = pathlib.Path(__file__).parent.resolve() / "results" / "nonconvex_low"
    solver_time = 483.14


data_fpi = jnp.load(path / filename_fpi)
data_bicgstab = jnp.load(path / filename_bicgstab)
data_unroll = jnp.load(path / filename_unroll)
data_dc3 = jnp.load(path / filename_dc3)
# %%
results_matrix = jnp.array(
    [
        [
            data_fpi["obj_fun_test"].mean().item(),
            jnp.mean(jnp.max(data_fpi["eq_viol_test"], axis=1), axis=0).item(),
            jnp.max(jnp.max(data_fpi["eq_viol_test"], axis=1), axis=0).item(),
            jnp.mean(jnp.max(data_fpi["ineq_viol_test"], axis=1), axis=0).item(),
            jnp.max(jnp.max(data_fpi["ineq_viol_test"], axis=1), axis=0).item(),
            data_fpi["inference_time"].item(),
            data_fpi["training_time"].item(),
        ],
        [
            data_bicgstab["obj_fun_test"].mean().item(),
            jnp.mean(jnp.max(data_bicgstab["eq_viol_test"], axis=1), axis=0).item(),
            jnp.max(jnp.max(data_bicgstab["eq_viol_test"], axis=1), axis=0).item(),
            jnp.mean(jnp.max(data_bicgstab["ineq_viol_test"], axis=1), axis=0).item(),
            jnp.max(jnp.max(data_bicgstab["ineq_viol_test"], axis=1), axis=0).item(),
            data_bicgstab["inference_time"].item(),
            data_bicgstab["training_time"].item(),
        ],
        [
            data_unroll["obj_fun_test"].mean().item(),
            jnp.mean(jnp.max(data_unroll["eq_viol_test"], axis=1), axis=0).item(),
            jnp.max(jnp.max(data_unroll["eq_viol_test"], axis=1), axis=0).item(),
            jnp.mean(jnp.max(data_unroll["ineq_viol_test"], axis=1), axis=0).item(),
            jnp.max(jnp.max(data_unroll["ineq_viol_test"], axis=1), axis=0).item(),
            data_unroll["inference_time"].item(),
            data_unroll["training_time"].item(),
        ],
        [
            data_dc3["mean_objective"].item(),
            data_dc3["mean_equality_max"].item(),
            data_dc3["max_equality_max"].item(),
            data_dc3["mean_inequality_max"].item(),
            data_dc3["max_inequality_max"].item(),
            data_dc3["inference_time"].item(),
            data_dc3["train_time"].item(),
        ],
        [
            data_fpi["opt_obj_test"].mean().item(),
            0.0,
            0.0,
            0.0,
            0.0,
            solver_time,
            -1.0,
        ],
    ]
)

# Define the row and column labels.
row_labels = ["FPI", "Indirect", "Unrolled", "DC3", "Optimizer"]
col_labels = [
    "Obj Fun Test",
    "Mean Eq Viol",
    "Max Eq Viol",
    "Mean Ineq Viol",
    "Max Ineq Viol",
    "Inference Time [s]",
    "Training Time [s]",
]

# Convert the results matrix to a Pandas DataFrame.
df = pd.DataFrame(results_matrix, index=row_labels, columns=col_labels)

# Create a copy for formatting.
formatted_df = df.copy()

for col in ["Mean Eq Viol", "Max Eq Viol", "Mean Ineq Viol", "Max Ineq Viol"]:
    formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2e}")

# Print the formatted DataFrame with vertical lines using tabulate.
print("")
print(f"======================== {label} ========================")
print("")
print(tabulate(formatted_df, headers="keys", tablefmt="grid", showindex=True))
# %%
