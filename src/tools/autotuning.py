# %%
"""Autotuning pipeline.

We recommend using this script as a Jupyter notebook.
To easily do so, run:
pip install jupytext
jupytext --set-formats ipynb,py:percent --sync src/hcnn/autotuning.py
"""

# %%
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmarks.QP.load_qp import (
    DC3Dataset,
    SimpleQPDataset,
    create_dataloaders,
    dc3_dataloader,
)
from pinet import (
    AffineInequalityConstraint,
    EqualityConstraint,
    EqualityConstraintsSpecification,
    EquilibrationParams,
    Project,
    ProjectionInstance,
)
from pinet._typing import (
    BatchedEqMatrix,
    BatchedIneqMatrix,
    BatchedPrimal,
    BatchedRHS,
    BatchedScalar,
)

jax.config.update("jax_enable_x64", True)

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# NOTE: Change this to the absolute path of your datasets directory.
dataset_dir = "absolute/path/to/datasets"


@dataclass(frozen=True)
class LoadedData:
    """Container for problem tensors, dataset inputs, and dataloaders.

    Attributes:
        filename: Dataset filename or filename prefix used to load the problem.
        q: Linear cost term of the quadratic program.
        p: Quadratic cost matrix of the quadratic program.
        a_dyn: Equality-constraint matrix.
        constr_matrix: Inequality-constraint matrix.
        h: Constraint right-hand side tensor.
        x_dataset: Input dataset associated with the loaded problem instance.
        train_loader: Dataloader for the training split.
        valid_loader: Dataloader for the validation split.
        test_loader: Dataloader for the test split.
    """

    filename: str
    q: jax.Array
    p: jax.Array
    a_dyn: BatchedEqMatrix
    constr_matrix: BatchedIneqMatrix
    h: jax.Array
    x_dataset: jax.Array
    train_loader: Any
    valid_loader: Any
    test_loader: Any


def load_data(
    use_dc3_dataset: bool,
    use_convex: bool,
    problem_seed: int,
    problem_var: int,
    problem_nineq: int,
    problem_neq: int,
    problem_examples: int,
) -> LoadedData:
    """Load problem data and dataloaders.

    Args:
        use_dc3_dataset: Whether to load the DC3 dataset layout.
        use_convex: Whether to load a convex problem instance.
        problem_seed: Random seed used in the simple QP dataset name.
        problem_var: Number of optimization variables.
        problem_nineq: Number of inequality constraints.
        problem_neq: Number of equality constraints.
        problem_examples: Number of examples in the dataset.

    Returns:
        LoadedData: Filename, problem tensors, dataset inputs, and
            train/validation/test dataloaders.

    Raises:
        NotImplementedError: If a non-convex simple QP dataset is requested.
    """
    if not use_dc3_dataset:
        # Choose problem parameters
        if use_convex:
            filename = (
                f"SimpleQP_seed{problem_seed}_var{problem_var}_ineq{problem_nineq}"
                f"_eq{problem_neq}_examples{problem_examples}.npz"
            )
        else:
            raise NotImplementedError()
        dataset_path = os.path.join(dataset_dir, filename)

        qp_dataset = SimpleQPDataset(dataset_path)
        train_loader, valid_loader, test_loader = create_dataloaders(
            dataset_path, batch_size=2048, val_split=0.1, test_split=0.1
        )
        q, p, a_dyn, constr_matrix, h = qp_dataset.const
        p = p[0, :, :]
        x_dataset = qp_dataset.x_data
    else:
        # Choose the filename here
        if use_convex:
            filename = (
                f"dc3_random_simple_dataset_var{problem_var}_ineq{problem_nineq}"
                f"_eq{problem_neq}_ex{problem_examples}"
            )
        else:
            filename = (
                f"dc3_random_nonconvex_dataset_var{problem_var}_ineq{problem_nineq}"
                f"_eq{problem_neq}_ex{problem_examples}"
            )
        filename_train = filename + "train.npz"
        dataset_path_train = os.path.join(dataset_dir, filename_train)
        filename_valid = filename + "valid.npz"
        dataset_path_valid = os.path.join(dataset_dir, filename_valid)
        filename_test = filename + "test.npz"
        dataset_path_test = os.path.join(dataset_dir, filename_test)
        train_loader = dc3_dataloader(dataset_path_train, use_convex, batch_size=2048)
        valid_loader = dc3_dataloader(
            dataset_path_valid, use_convex, batch_size=1024, shuffle=False
        )
        test_loader = dc3_dataloader(
            dataset_path_test, use_convex, batch_size=1024, shuffle=False
        )
        dataset = cast(
            DC3Dataset,
            cast(DataLoader[tuple[jax.Array, jax.Array]], train_loader).dataset,
        )
        q, p, a_dyn, constr_matrix, h = dataset.const
        p = p[0, :, :]
        x_dataset = dataset.x_data

    return LoadedData(
        filename=filename,
        q=q,
        p=p,
        a_dyn=a_dyn,
        constr_matrix=constr_matrix,
        h=h,
        x_dataset=x_dataset,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
    )


# %%
# Load a batch of data for autotuning
data = load_data(
    use_dc3_dataset=True,
    use_convex=True,
    problem_seed=42,
    problem_var=1000,
    problem_nineq=500,
    problem_neq=500,
    problem_examples=10000,
)
filename = data.filename
q = data.q
p = data.p
a_dyn = data.a_dyn
constr_matrix = data.constr_matrix
h = data.h
x_dataset = data.x_dataset
train_loader = data.train_loader
valid_loader = data.valid_loader
test_loader = data.test_loader
n_samples = 150
x_batch, _ = next(iter(valid_loader))
x_batch = x_batch[:n_samples]


# %%


def build_evaluate_params(
    x: BatchedPrimal,
    b: BatchedRHS,
    n_iter: int,
    project: Callable[..., tuple[ProjectionInstance, jax.Array]],
    compute_cv: Callable[[ProjectionInstance], jax.Array],
) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]]:
    """Build an evaluator for autotuning hyperparameters.

    Args:
        x: Batch of points to project.
        b: Batch of equality-constraint right-hand sides.
        n_iter: Number of projection iterations to apply per evaluation.
        project: Projection function returning the projected instance and next state.
        compute_cv: Function computing constraint violation for a projection result.

    Returns:
        Function mapping ``(init, sigma)`` to the next warm start, maximum
        constraint violation, and mean projection distance.
    """

    def evaluate_params(init, sigma):
        y, init = project(init=init, x=x, b=b, sigma=sigma, n_iter=n_iter)
        cvs = compute_cv(y)
        values = jnp.linalg.norm(y.x - x)
        return init, jnp.max(cvs), jnp.mean(values)

    return evaluate_params


# %%
# Setup the projection layer
eq_constraint = EqualityConstraint(a_dyn=a_dyn, b=x_batch, method=None, var_b=True)
ineq_constraint = AffineInequalityConstraint(
    constr_matrix=constr_matrix,
    ub=h,
    lb=-jnp.inf * jnp.ones_like(h),
)
projection_layer = Project(
    ineq_constraint=ineq_constraint,
    eq_constraint=eq_constraint,
    unroll=False,
    equilibration_params=EquilibrationParams(
        max_iter=25,
        tol=1.0e-3,
        ord=2.0,
        col_scaling=False,
        update_mode="Gauss",
        safeguard=False,
    ),
)

# %%
omega = 1.7


def project(
    init: jax.Array,
    x: BatchedPrimal,
    b: BatchedRHS,
    sigma: BatchedScalar | float,
    n_iter: int,
) -> tuple[ProjectionInstance, jax.Array]:
    """Wrap the projection layer call.

    Args:
        init: Initial solver state.
        x: Batch of points to project.
        b: Batch of equality-constraint right-hand sides.
        sigma: Penalty parameter used by the projection layer.
        n_iter: Number of iterations to run.

    Returns:
        Projected instance and updated solver state.
    """
    yraw = ProjectionInstance(x=x, eq=EqualityConstraintsSpecification(b=b))
    return projection_layer.call(
        s0=init,
        yraw=yraw,
        sigma=sigma,
        omega=omega,
        n_iter=n_iter,
    )


def compute_cv(y: ProjectionInstance) -> jax.Array:
    """Compute constraint violation.

    Args:
        y: Projected instance to evaluate.

    Returns:
        Flattened constraint-violation values.
    """
    # ``projection_layer.cv`` returns ``BatchedScalar`` (ArrayLike) at the
    # public boundary; coerce to ``jax.Array`` for the script's downstream
    # use which expects jax-only attributes (``.reshape``, ``.at`` ...).
    return jnp.asarray(projection_layer.cv(y)).reshape(-1)


x = jax.random.normal(
    jax.random.PRNGKey(0), (x_batch.shape[0], projection_layer.dim, 1)
)  # batch of random points to project

# %%
# Target values for sigma tuning
target_cv_sigma = 5e-2
target_rs_sigma = 1e-1
# Target values for n_iter tuning
target_cv_n_iter = 1e-3
target_rs_n_iter = 1e-2
# Fixed n_iter for the first stage
fixed_n_max_iter = 100
fixed_n_iter_step = 100
fixed_n_iter_candidates = fixed_n_max_iter // fixed_n_iter_step
# n_iter candidates for the second stage
n_max_iter = 400
n_iter_step = 50
n_iter_candidates = n_max_iter // n_iter_step

tie_breaker = "cv"
if tie_breaker == "cv":
    id_tie_breaker = 0
elif tie_breaker == "rs":
    id_tie_breaker = 1

sigma_candidates = jnp.logspace(-3, jnp.log10(5.05), num=100)

init_shape = (x_batch.shape[0], projection_layer.dim_lifted, 1)

# Evaluate the first stage
fixed_eval_fn = jax.jit(
    build_evaluate_params(x, x_batch, fixed_n_iter_step, project, compute_cv)
)
# Evaluate the second stage
eval_fn = jax.jit(build_evaluate_params(x, x_batch, n_iter_step, project, compute_cv))


# %%
def generate_results(
    sigma_candidates: Float[Array, "n_sigma"],
    n_iter_candidates: int,
    eval_fn: Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]],
) -> Float[Array, "n_sigma n_iter_candidates 2"]:
    """Evaluate candidate hyperparameters on the validation batch.

    Args:
        sigma_candidates: Candidate sigma values to test.
        n_iter_candidates: Number of iteration-count candidates to evaluate.
        eval_fn: Evaluation function returned by ``build_evaluate_params``.

    Returns:
        Array of shape ``(n_sigma, n_iter_candidates, 2)`` containing the maximum
        constraint violation and mean projection distance.
    """
    # Initialize results array
    results = jnp.inf * jnp.ones((len(sigma_candidates), n_iter_candidates, 2))

    def body_fun(i, r):
        sigma = sigma_candidates[i]

        def body_fun_i(j, state_i):
            ri, init = state_i
            init, cv, val = eval_fn(init, sigma)
            return ri.at[j, :].set(jnp.stack([cv, val])), init

        init = jnp.zeros(init_shape)
        _r, _ = jax.lax.fori_loop(0, n_iter_candidates, body_fun_i, (r[i, ...], init))
        return r.at[i, ...].set(_r)

    # Wrap the range with tqdm to display a progress bar
    for i in tqdm(range(len(sigma_candidates)), desc="Processing candidates"):
        results = body_fun(i, results)

    return results


# %%
def get_best(
    results: Float[Array, "n_sigma n_iter_candidates 2"],
    sigma_candidates: Float[Array, "n_sigma"],
    n_iter_step: int,
    target_cv: float,
    target_rs: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Select the best hyperparameter combination satisfying target metrics.

    Args:
        results: Evaluation results containing constraint violation and objective gap.
        sigma_candidates: Candidate sigma values corresponding to ``results`` rows.
        n_iter_step: Iteration increment represented by one column in ``results``.
        target_cv: Maximum acceptable constraint violation.
        target_rs: Maximum acceptable relative suboptimality.

    Returns:
        Best sigma, best iteration count, and the corresponding ``[cv, value]``
        result pair.

    Raises:
        ValueError: If no candidate pair satisfies the target conditions.
    """
    # Use the best result as proxy for the optimal value
    opt = jnp.min(results[:, :, 1])
    # Compute the relative suboptimality
    rs = (results[:, :, 1] - opt) / (opt + 1e-20)
    # Compute which entries satisfy both target conditions
    mask = (results[:, :, 0] < target_cv) * (rs < target_rs)
    mask_valid_sigma = jnp.any(mask, axis=1)
    if jnp.sum(mask) == 0:
        raise ValueError("No valid sigma found for the given target conditions.")
    # For each row, find the first column index where the condition is met
    first_valid_idx = jnp.argmax(mask, axis=1)
    # Find the minimum number of iterations across all rows
    min_iter_idx = jnp.min(first_valid_idx[mask_valid_sigma])
    # Find the best sigma values
    best_sigma_mask = (first_valid_idx == min_iter_idx) * mask_valid_sigma

    if jnp.sum(best_sigma_mask) > 1:
        # Tie breaking
        mask = mask * best_sigma_mask[:, None]
        min_val = jnp.min(results[mask, id_tie_breaker])
        best_sigma_mask = best_sigma_mask & jnp.any(
            results[..., id_tie_breaker] == min_val, axis=1
        )
        if jnp.sum(best_sigma_mask) > 1:
            # Other tie breaking
            mask = mask * best_sigma_mask[:, None]
            min_val = jnp.min(results[mask, 1 - id_tie_breaker])
            best_sigma_mask = best_sigma_mask & jnp.any(
                results[..., id_tie_breaker] == min_val, axis=1
            )

    # Find the (first) index non-zero in best_sigma_mask
    best_sigma_idx = jnp.argmax(best_sigma_mask)
    best_sigma = sigma_candidates[best_sigma_idx]
    best_n_iter = n_iter_step * (min_iter_idx + 1)

    return best_sigma, best_n_iter, results[best_sigma_idx, min_iter_idx, :]


# %%
results_sigma = generate_results(sigma_candidates, fixed_n_iter_candidates, fixed_eval_fn)

# %%
print("=========== Results for fixed n_iter ===========")
best_sigma, best_n_iter, best_result = get_best(
    results_sigma, sigma_candidates, fixed_n_iter_step, target_cv_sigma, target_rs_sigma
)
print(f"Best sigma: {best_sigma}")
best_sigma = jnp.array([best_sigma])

# %%
results_n_iter = generate_results(best_sigma, n_iter_candidates, eval_fn)

# %%
print("=========== Results for n_iter tuning ===========")
best_sigma, best_n_iter, best_result = get_best(
    results_n_iter, best_sigma, n_iter_step, target_cv_n_iter, target_rs_n_iter
)
print(f"Best sigma: {best_sigma}")
print(f"Best n_iter: {best_n_iter}")
