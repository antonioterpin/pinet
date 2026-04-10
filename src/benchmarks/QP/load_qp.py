"""Loading functionality for simple QP benchmark."""

import os
from collections.abc import Callable, Iterator
from typing import cast

import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader, Dataset, random_split


# Load Instance Dataset
class SimpleQPDataset(Dataset):
    """Dataset for simple QP benchmark."""

    def __init__(self, filepath: str) -> None:
        """Initialize dataset.

        Args:
            filepath: Path to the dataset file.
        """
        data = jnp.load(filepath)
        # Parameter values for each instance
        self.x_data = data["X"]
        # Constant problem ingredients
        self.const = (data["Q"], data["p"], data["A"], data["G"], data["h"])
        # Optimal objectives and solutions for all problem instances
        self.objectives = data["objectives"]
        self.ystar = data["Ystar"]

    def __len__(self) -> int:
        """Length of dataset.

        Returns:
            Number of items in the dataset.
        """
        return self.x_data.shape[0]

    def __getitem__(self, idx: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get item from dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Input features and objective value for the item.
        """
        return self.x_data[idx], self.objectives[idx]


def create_dataloaders(
    filepath: str,
    batch_size: int = 512,
    val_split: float = 0.0,
    test_split: float = 0.1,
    shuffle: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Dataset loaders for training, validation and test.

    Args:
        filepath: Path to the dataset file.
        batch_size: Size of each batch.
        val_split: Proportion of data to use for validation.
        test_split: Proportion of data to use for testing.
        shuffle: Whether to shuffle the dataset.

    Returns:
        DataLoaders for training, validation, and test datasets.
    """
    dataset = SimpleQPDataset(filepath)
    size = len(dataset)

    val_size = int(size * val_split)
    test_size = int(size * test_split)
    train_size = size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    def collate_fn(batch):
        x_data, obj = zip(*batch, strict=False)
        return jnp.array(x_data), jnp.array(obj)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=val_size, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, shuffle=False, batch_size=test_size, collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


class DC3Dataset(Dataset):
    """Dataset for importing DC3 problems."""

    def __init__(self, filepath: str, use_convex: bool):
        """Initialize dataset.

        Args:
            filepath: Path to the dataset file.
            use_convex: Whether to use convex problems.
        """
        data = jnp.load(filepath)
        # Parameter values for each instance
        self.x_data = data["X"]
        # Constant problem ingredients
        self.const = (data["Q"], data["p"], data["A"], data["G"], data["h"])
        # Problem solutions
        self.ystar = data["Ystar"]

        # Compute objectives
        if use_convex:

            def obj_fun(y):
                return 0.5 * y.T @ data["Q"] @ y + data["p"][0, :, :].T @ y

        else:

            def obj_fun(y):
                return 0.5 * y.T @ data["Q"] @ y + data["p"][0, :, :].T @ jnp.sin(y)

        self.obj_fun = jax.vmap(obj_fun, in_axes=[0])
        self.objectives = self.obj_fun(self.ystar[:, :, 0])

    def __len__(self) -> int:
        """Length of dataset.

        Returns:
            Number of items in the dataset.
        """
        return self.x_data.shape[0]

    def __getitem__(self, idx: int | jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get item from dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Input features and corresponding objective value.
        """
        return self.x_data[idx], self.objectives[idx]


class JaxDataLoader:
    """Dataloader for DC3 dataset implemented in JAX."""

    def __init__(
        self,
        filepath: str,
        use_convex: bool,
        batch_size: int,
        shuffle: bool = True,
        rng_key: jax.Array | None = None,
    ) -> None:
        """Initialize JaxDataLoader.

        Args:
            filepath: Path to the dataset file.
            use_convex: Whether to use convex problems.
            batch_size: Size of each batch.
            shuffle: Whether to shuffle the dataset.
            rng_key: Random key for shuffling.
        """
        self.dataset = DC3Dataset(filepath, use_convex)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        # Batch indices for the current epoch
        self._perm = self._get_perm() if self.shuffle else jnp.arange(len(self.dataset))

    def __len__(self) -> int:
        """Length of dataset.

        Returns:
            Number of batches in the dataset.
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[jnp.ndarray, jnp.ndarray]]:
        """Iterate over the dataset.

        Yields:
            tuple[jnp.ndarray, jnp.ndarray]: Batch of input features and objective values.
        """
        for start in range(0, len(self.dataset), self.batch_size):
            batch_idx = self._perm[start : start + self.batch_size]
            yield self.dataset[batch_idx]

        if self.shuffle:
            self._perm = self._get_perm()

    def _get_perm(self) -> jax.Array:
        self.rng_key, last_key = jax.random.split(self._rng_key)
        perm = jax.random.permutation(last_key, len(self.dataset))
        return perm


def dc3_dataloader(
    filepath: str,
    use_convex: bool,
    batch_size: int = 512,
    shuffle: bool = True,
) -> DataLoader:
    """Dataset loader for training, validation, or test.

    Args:
        filepath: Path to the dataset file.
        use_convex: Whether to use convex problems.
        batch_size: Size of each batch.
        shuffle: Whether to shuffle the dataset.

    Returns:
        DataLoader for the dataset.
    """
    dataset = DC3Dataset(filepath, use_convex)

    def collate_fn(batch):
        x_data, obj = zip(*batch, strict=False)
        return jnp.array(x_data), jnp.array(obj)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )

    return loader


def non_dc3_dataset_setup(
    use_convex: bool,
    problem_seed: int,
    problem_var: int,
    problem_nineq: int,
    problem_neq: int,
    problem_examples: int,
    rng_key: jax.Array,
    batch_size: int,
    use_jax_loader: bool,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """Setup function for datasets generated with our script.

    Args:
        use_convex: Whether to use convex problems.
        problem_seed: Seed for random number generation.
        problem_var: Variance of the problem.
        problem_nineq: Number of inequality constraints.
        problem_neq: Number of equality constraints.
        problem_examples: Number of examples in the dataset.
        rng_key: Random key for JAX operations.
            Unused in this function but kept for consistency with other loaders.
        batch_size: Size of each batch.
        use_jax_loader: Whether to use JAX DataLoader.
            Unused in this function but kept for consistency with other loaders.

    Returns:
        Problem matrices, input features, and train, validation, and test loaders.

    Raises:
        NotImplementedError: If non-convex generated datasets are requested.
    """
    # Choose problem parameters
    if use_convex:
        filename = (
            f"SimpleQP_seed{problem_seed}_var{problem_var}_ineq{problem_nineq}"
            f"_eq{problem_neq}_examples{problem_examples}.npz"
        )
    else:
        raise NotImplementedError()
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", filename)

    qp_dataset = SimpleQPDataset(dataset_path)
    train_loader, valid_loader, test_loader = create_dataloaders(
        dataset_path, batch_size=batch_size, val_split=0.1, test_split=0.1
    )
    q_mat, p, a_dyn, g_mat, h = qp_dataset.const
    p = p[0, :, :]
    x_data = qp_dataset.x_data

    return q_mat, p, a_dyn, g_mat, h, x_data, train_loader, valid_loader, test_loader


def dc3_dataset_setup(
    use_convex: bool,
    problem_seed: int,
    problem_var: int,
    problem_nineq: int,
    problem_neq: int,
    problem_examples: int,
    rng_key: jax.Array,
    batch_size: int,
    use_jax_loader: bool,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    DataLoader | JaxDataLoader,
    DataLoader | JaxDataLoader,
    DataLoader | JaxDataLoader,
]:
    """Setup function for datasets generated with the DC3 script.

    Args:
        use_convex: Whether to use convex problems.
        problem_seed: Seed for random number generation.
        problem_var: Variance of the problem.
        problem_nineq: Number of inequality constraints.
        problem_neq: Number of equality constraints.
        problem_examples: Number of examples in the dataset.
        rng_key: Random key for JAX operations.
        batch_size: Size of each batch.
        use_jax_loader: Whether to use JAX DataLoader.

    Returns:
        Problem matrices, input features, and train, validation, and test loaders.
    """
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
    dataset_path_train = os.path.join(
        os.path.dirname(__file__), "datasets", filename_train
    )
    filename_valid = filename + "valid.npz"
    dataset_path_valid = os.path.join(
        os.path.dirname(__file__), "datasets", filename_valid
    )
    filename_test = filename + "test.npz"
    dataset_path_test = os.path.join(os.path.dirname(__file__), "datasets", filename_test)
    if not use_jax_loader:
        train_loader = dc3_dataloader(
            dataset_path_train, use_convex, batch_size=batch_size
        )
        valid_loader = dc3_dataloader(
            dataset_path_valid, use_convex, batch_size=1024, shuffle=False
        )
        test_loader = dc3_dataloader(
            dataset_path_test, use_convex, batch_size=1024, shuffle=False
        )
    else:
        loader_keys = jax.random.split(rng_key, 3)
        train_loader = JaxDataLoader(
            dataset_path_train,
            use_convex,
            batch_size=batch_size,
            rng_key=loader_keys[0],
        )
        valid_loader = JaxDataLoader(
            dataset_path_valid,
            use_convex,
            batch_size=1024,
            shuffle=False,
            rng_key=loader_keys[1],
        )
        test_loader = JaxDataLoader(
            dataset_path_test,
            use_convex,
            batch_size=1024,
            shuffle=False,
            rng_key=loader_keys[2],
        )
    dataset = cast(DC3Dataset, train_loader.dataset)
    q_mat, p, a_dyn, g_mat, h = dataset.const
    p = p[0, :, :]
    x_data = dataset.x_data

    return q_mat, p, a_dyn, g_mat, h, x_data, train_loader, valid_loader, test_loader


def load_data(
    use_dc3_dataset: bool,
    use_convex: bool,
    problem_seed: int,
    problem_var: int,
    problem_nineq: int,
    problem_neq: int,
    problem_examples: int,
    rng_key: jax.Array,
    batch_size: int = 2048,
    use_jax_loader: bool = True,
    penalty: float = 0.0,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    Callable[[jnp.ndarray], jnp.ndarray],
    DataLoader | JaxDataLoader,
    DataLoader | JaxDataLoader,
    DataLoader | JaxDataLoader,
    Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
]:
    """Load problem data.

    Args:
        use_dc3_dataset: Whether to use the DC3 dataset.
        use_convex: Whether to use convex problems.
        problem_seed: Seed for random number generation.
        problem_var: Variance of the problem.
        problem_nineq: Number of inequality constraints.
        problem_neq: Number of equality constraints.
        problem_examples: Number of examples in the dataset.
        rng_key: Random key for JAX operations.
        batch_size: Size of each batch.
        use_jax_loader: Whether to use JAX DataLoader.
        penalty: Penalty term for the loss function.

    Returns:
        Constraint matrices, input features, loaders, and objective/loss functions.

    """
    if not use_dc3_dataset:
        setup = non_dc3_dataset_setup
    else:
        setup = dc3_dataset_setup

    q_mat, p, a_dyn, g_mat, h, x_data, train_loader, valid_loader, test_loader = setup(
        use_convex=use_convex,
        problem_seed=problem_seed,
        problem_var=problem_var,
        problem_nineq=problem_nineq,
        problem_neq=problem_neq,
        problem_examples=problem_examples,
        rng_key=rng_key,
        batch_size=batch_size,
        use_jax_loader=use_jax_loader,
    )

    # Define loss/objective function
    # Predictions is of shape (batch_size, Y_DIM) and Q is of shape (Y_DIM, Y_DIM)
    def quadratic_form(prediction: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the quadratic objective.

        Args:
            prediction: Predicted solution.

        Returns:
            Objective value for the prediction.
        """
        return 0.5 * prediction.T @ q_mat @ prediction + p.T @ prediction

    def quadratic_form_sine(prediction: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the quadratic objective plus sine.

        Args:
            prediction: Predicted solution.

        Returns:
            Objective value for the prediction.
        """
        return 0.5 * prediction.T @ q_mat @ prediction + p.T @ jnp.sin(prediction)

    if use_convex:
        objective_function = quadratic_form
    else:
        objective_function = quadratic_form_sine

    # Vectorize the quadratic form computation over the batch dimension
    batched_objective = jax.vmap(objective_function, in_axes=[0])

    def penalty_form(predictions, x_data):
        eq_cv = jnp.max(
            jnp.abs(
                a_dyn[0].reshape(1, a_dyn.shape[1], a_dyn.shape[2])
                @ predictions.reshape(x_data.shape[0], a_dyn.shape[2], 1)
                - x_data
            ),
            axis=1,
        )
        ineq_cv = jnp.max(
            jnp.maximum(
                g_mat[0].reshape(1, g_mat.shape[1], g_mat.shape[2])
                @ predictions.reshape(x_data.shape[0], g_mat.shape[2], 1)
                - h,
                0,
            ),
            axis=1,
        )

        return eq_cv + ineq_cv

    def batched_loss(predictions, x_data):
        if penalty > 0:
            return batched_objective(predictions) + penalty * penalty_form(
                predictions, x_data
            )
        else:
            return batched_objective(predictions)

    return (
        a_dyn,
        g_mat,
        h,
        x_data,
        batched_objective,
        train_loader,
        valid_loader,
        test_loader,
        batched_loss,
    )
