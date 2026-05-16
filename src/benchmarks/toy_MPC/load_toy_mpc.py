"""Loading functionality for toy MPC benchmark."""

import os
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader, Dataset, random_split


# Load Instance Dataset
class ToyMPCDataset(Dataset[tuple[jax.Array, jax.Array]]):
    """Dataset for toy MPC benchmark."""

    def __init__(self, data: dict[str, jax.Array], const: dict[str, jax.Array]):
        """Initialize dataset.

        Args:
            data: Dictionary containing the dataset.
            const: Dictionary containing constant problem ingredients.
        """
        # Parameter values for each instance
        self.x0sets = data["x0sets"]
        # Constant problem ingredients
        self.const = (
            const["a_mat"],
            const["lbxs"],
            const["ubxs"],
            const["lbus"],
            const["ubus"],
            const["xhat"],
            const["alpha"],
            const["horizon"],
            const["base_dim"],
        )
        # Optimal objectives and solutions for all problem instances
        self.objectives = data["objectives"]
        self.y_star = data["y_star"]

    def __len__(self) -> int:
        """Length of dataset.

        Returns:
            int: Number of instances in the dataset.
        """
        return self.x0sets.shape[0]

    def __getitem__(self, idx: int | jax.Array) -> tuple[jax.Array, jax.Array]:
        """Get item from dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            tuple[jax.Array, jax.Array]:
                Tuple containing the initial condition and the objective value.
        """
        return self.x0sets[idx], self.objectives[idx]


def create_dataloaders(
    dataset: ToyMPCDataset,
    batch_size: int = 2048,
    val_split: float = 0.1,
    test_split: float = 0.1,
    shuffle: bool = True,
) -> tuple[
    DataLoader[tuple[jax.Array, jax.Array]],
    DataLoader[tuple[jax.Array, jax.Array]],
    DataLoader[tuple[jax.Array, jax.Array]],
]:
    """Dataset loaders for training, validation and test.

    Args:
        dataset: The dataset to create loaders for.
        batch_size: Size of each batch.
        val_split: Proportion of the dataset to use for validation.
        test_split: Proportion of the dataset to use for testing.
        shuffle: Whether to shuffle the training dataloader.

    Returns:
        A tuple containing the training, validation, and test DataLoaders.
    """
    size = len(dataset)

    val_size = int(size * val_split)
    test_size = int(size * test_split)
    train_size = size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    def collate_fn(batch):
        x0sets, obj = zip(*batch, strict=False)
        return jnp.array(x0sets), jnp.array(obj)

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


class JaxDataLoaderMPC:
    """Dataloader for toy MPC dataset in JAX.

    Standalone loader (does not inherit from ``JaxDataLoader``) because the MPC
    loader takes a prebuilt dataset instead of a filepath.
    """

    def __init__(
        self,
        dataset: ToyMPCDataset,
        batch_size: int,
        shuffle: bool = True,
        rng_key: jax.Array | None = None,
    ) -> None:
        """Initialize loader.

        Args:
            dataset: The dataset to load.
            batch_size: Size of each batch.
            shuffle: Whether to shuffle the dataset.
            rng_key: Random key for shuffling.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        # Batch indices for the current epoch
        self._perm = self._get_perm() if self.shuffle else jnp.arange(len(self.dataset))

    def __len__(self) -> int:
        """Number of batches per epoch.

        Returns:
            Number of batches.
        """
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Iterate over batches of the dataset.

        Yields:
            Tuple ``(x, y)`` for each batch.
        """
        for start in range(0, len(self.dataset), self.batch_size):
            batch_idx = self._perm[start : start + self.batch_size]
            yield self.dataset[batch_idx]
        if self.shuffle:
            self._perm = self._get_perm()

    def _get_perm(self) -> jax.Array:
        self._rng_key, last_key = jax.random.split(self._rng_key)
        return jax.random.permutation(last_key, len(self.dataset))


def load_data(
    filepath: str,
    rng_key: jax.Array,
    batch_size: int = 2048,
    val_split: float = 0.1,
    test_split: float = 0.1,
    use_jax_loader: bool = True,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    float,
    int,
    int,
    jax.Array,
    DataLoader[tuple[jax.Array, jax.Array]] | JaxDataLoaderMPC,
    DataLoader[tuple[jax.Array, jax.Array]] | JaxDataLoaderMPC,
    DataLoader[tuple[jax.Array, jax.Array]] | JaxDataLoaderMPC,
    Callable[[jax.Array], jax.Array],
]:
    """Load problem data.

    Args:
        filepath: Path to the dataset file.
        rng_key: Random key for shuffling.
        batch_size: Size of each batch.
        val_split: Proportion of the dataset to use for validation.
        test_split: Proportion of the dataset to use for testing.
        use_jax_loader: Whether to use JAX DataLoader or PyTorch DataLoader.

    Returns:
        tuple: A tuple containing:
            - As (jax.Array): System dynamics matrix.
            - lbxs (jax.Array): Lower bounds for state variables.
            - ubxs (jax.Array): Upper bounds for state variables.
            - lbus (jax.Array): Lower bounds for control inputs.
            - ubus (jax.Array): Upper bounds for control inputs.
            - xhat (jax.Array): Reference state.
            - alpha (float): Regularization parameter.
            - T (int): Time horizon.
            - base_dim (int): Dimension of the base state.
            - X (jax.Array): Initial conditions for the dataset.
            - train_loader: Training DataLoader or JaxDataLoaderMPC.
            - valid_loader: Validation DataLoader or JaxDataLoaderMPC.
            - test_loader: Test DataLoader or JaxDataLoaderMPC.
            - batched_objective: Function to compute the quadratic objective in batches.
    """
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", filepath)
    all_data = cast(dict[str, jax.Array], cast(object, jnp.load(dataset_path)))
    toy_dataset = ToyMPCDataset(all_data, all_data)
    if not use_jax_loader:
        train_loader, valid_loader, test_loader = create_dataloaders(
            dataset=toy_dataset,
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split,
        )
    else:
        total_size = all_data["x0sets"].shape[0]
        val_size = int(val_split * total_size)
        test_size = int(test_split * total_size)
        train_size = total_size - val_size - test_size

        perm_key, rng_key = jax.random.split(rng_key, 2)
        permutation = jax.random.permutation(perm_key, total_size)
        train_idx = permutation[:train_size]
        val_idx = permutation[train_size : train_size + val_size]
        test_idx = permutation[train_size + val_size :]

        y_star_full = all_data["y_star"]
        train_dataset = {
            "x0sets": all_data["x0sets"][train_idx],
            "objectives": all_data["objectives"][train_idx],
            "y_star": y_star_full[train_idx],
        }
        train_dataset = ToyMPCDataset(train_dataset, all_data)
        val_dataset = {
            "x0sets": all_data["x0sets"][val_idx],
            "objectives": all_data["objectives"][val_idx],
            "y_star": y_star_full[val_idx],
        }
        val_dataset = ToyMPCDataset(val_dataset, all_data)
        test_dataset = {
            "x0sets": all_data["x0sets"][test_idx],
            "objectives": all_data["objectives"][test_idx],
            "y_star": y_star_full[test_idx],
        }
        test_dataset = ToyMPCDataset(test_dataset, all_data)

        loader_keys = jax.random.split(rng_key, 3)
        train_loader = JaxDataLoaderMPC(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            rng_key=loader_keys[0],
        )
        valid_loader = JaxDataLoaderMPC(
            dataset=val_dataset,
            batch_size=val_size,
            shuffle=False,
            rng_key=loader_keys[1],
        )
        test_loader = JaxDataLoaderMPC(
            dataset=test_dataset,
            batch_size=test_size,
            shuffle=False,
            rng_key=loader_keys[2],
        )

    a, lbxs, ubxs, lbus, ubus, xhat, alpha_raw, horizon_raw, base_dim_raw = (
        toy_dataset.const
    )
    # Scalar metadata was stored as 0-d arrays; unwrap to Python scalars.
    alpha = float(alpha_raw)
    horizon = int(horizon_raw)
    base_dim = int(base_dim_raw)
    x_data = toy_dataset.x0sets
    dimx = lbxs.shape[1]

    def quadratic_form(prediction: jax.Array) -> jax.Array:
        """Evaluate the quadratic objective.

        Args:
            prediction: Predicted trajectory and controls for one instance.

        Returns:
            jax.Array: Scalar quadratic objective value.
        """
        return jnp.sum(
            (prediction[:dimx] - jnp.tile(xhat[:, 0], horizon + 1)) ** 2
        ) + alpha * jnp.sum(prediction[dimx:] ** 2)

    batched_objective = jax.vmap(quadratic_form, in_axes=[0])

    return (
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
    )
