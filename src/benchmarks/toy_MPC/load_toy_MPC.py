"""Loading functionality for toy MPC benchmark."""

import os
from typing import Optional

import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader, Dataset, random_split

from benchmarks.QP.load_QP import JaxDataLoader


# Load Instance Dataset
class ToyMPCDataset(Dataset):
    """Dataset for toy MPC benchmark."""

    def __init__(self, data, const):
        """Initialize dataset."""
        # Parameter values for each instance
        self.x0sets = data["x0sets"]
        # Constant problem ingredients
        self.const = (
            const["As"],
            const["lbxs"],
            const["ubxs"],
            const["lbus"],
            const["ubus"],
            const["xhat"],
            const["alpha"],
            const["T"],
            const["base_dim"],
        )
        # Optimal objectives and solutions for all problem instances
        self.objectives = data["objectives"]
        self.Ystar = data["Ystar"]

    def __len__(self):
        """Length of dataset."""
        return self.x0sets.shape[0]

    def __getitem__(self, idx):
        """Get item from dataset."""
        return self.x0sets[idx], self.objectives[idx]


def create_dataloaders(
    dataset, batch_size=2048, val_split=0.1, test_split=0.1, shuffle=True
):
    """Dataset loaders for training, validation and test."""
    size = len(dataset)

    val_size = int(size * val_split)
    test_size = int(size * test_split)
    train_size = size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    def collate_fn(batch):
        x0sets, obj = zip(*batch)
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


class JaxDataLoaderMPC(JaxDataLoader):
    """Dataloader for toy MPC dataset in JAX."""

    def __init__(
        self,
        dataset: ToyMPCDataset,
        batch_size: int,
        shuffle: bool = True,
        rng_key: Optional[jax.Array] = None,
    ):
        """Initialize loader."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        # Batch indices for the current epoch
        self._perm = self._get_perm() if self.shuffle else jnp.arange(len(self.dataset))


def load_data(
    filepath,
    rng_key,
    batch_size=2048,
    val_split=0.1,
    test_split=0.1,
    use_jax_loader=True,
):
    """Load problem data."""
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", filepath)
    all_data = jnp.load(dataset_path)
    ToyDataset = ToyMPCDataset(all_data, all_data)
    if not use_jax_loader:
        train_loader, valid_loader, test_loader = create_dataloaders(
            dataset=ToyDataset,
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

        train_dataset = {
            "x0sets": all_data["x0sets"][train_idx],
            "objectives": all_data["objectives"][train_idx],
            "Ystar": all_data["Ystar"][train_idx],
        }
        train_dataset = ToyMPCDataset(train_dataset, all_data)
        val_dataset = {
            "x0sets": all_data["x0sets"][val_idx],
            "objectives": all_data["objectives"][val_idx],
            "Ystar": all_data["Ystar"][val_idx],
        }
        val_dataset = ToyMPCDataset(val_dataset, all_data)
        test_dataset = {
            "x0sets": all_data["x0sets"][test_idx],
            "objectives": all_data["objectives"][test_idx],
            "Ystar": all_data["Ystar"][test_idx],
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

    As, lbxs, ubxs, lbus, ubus, xhat, alpha, T, base_dim = ToyDataset.const
    X = ToyDataset.x0sets
    dimx = lbxs.shape[1]

    def quadratic_form(prediction):
        """Evaluate the quadratic objective."""
        return jnp.sum(
            (prediction[:dimx] - jnp.tile(xhat[:, 0], T + 1)) ** 2
        ) + alpha * jnp.sum(prediction[dimx:] ** 2)

    batched_objective = jax.vmap(quadratic_form, in_axes=[0])

    return (
        As,
        lbxs,
        ubxs,
        lbus,
        ubus,
        xhat,
        alpha,
        T,
        base_dim,
        X,
        train_loader,
        valid_loader,
        test_loader,
        batched_objective,
    )
