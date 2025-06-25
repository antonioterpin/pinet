"""Loading functionality for simple QP benchmark."""

import os
from typing import Optional

import jax
import jax.numpy as jnp
from torch.utils.data import DataLoader, Dataset, random_split


# Load Instance Dataset
class SimpleQPDataset(Dataset):
    """Dataset for simple QP benchmark."""

    def __init__(self, filepath):
        """Initialize dataset."""
        data = jnp.load(filepath)
        # Parameter values for each instance
        self.X = data["X"]
        # Constant problem ingredients
        self.const = (data["Q"], data["p"], data["A"], data["G"], data["h"])
        # Optimal objectives and solutions for all problem instances
        self.objectives = data["objectives"]
        self.Ystar = data["Ystar"]

    def __len__(self):
        """Length of dataset."""
        return self.X.shape[0]

    def __getitem__(self, idx):
        """Get item from dataset."""
        return self.X[idx], self.objectives[idx]


def create_dataloaders(
    filepath, batch_size=512, val_split=0.0, test_split=0.1, shuffle=True
):
    """Dataset loaders for training, validation and test."""
    dataset = SimpleQPDataset(filepath)
    size = len(dataset)

    val_size = int(size * val_split)
    test_size = int(size * test_split)
    train_size = size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    def collate_fn(batch):
        X, obj = zip(*batch)
        return jnp.array(X), jnp.array(obj)

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

    def __init__(self, filepath, use_convex):
        """Initialize dataset."""
        data = jnp.load(filepath)
        # Parameter values for each instance
        self.X = data["X"]
        # Constant problem ingredients
        self.const = (data["Q"], data["p"], data["A"], data["G"], data["h"])
        # Problem solutions
        self.Ystar = data["Ystar"]

        # Compute objectives
        if use_convex:

            def obj_fun(y):
                return 0.5 * y.T @ data["Q"] @ y + data["p"][0, :, :].T @ y

        else:

            def obj_fun(y):
                return 0.5 * y.T @ data["Q"] @ y + data["p"][0, :, :].T @ jnp.sin(y)

        self.obj_fun = jax.vmap(obj_fun, in_axes=[0])
        self.objectives = self.obj_fun(self.Ystar[:, :, 0])

    def __len__(self):
        """Length of dataset."""
        return self.X.shape[0]

    def __getitem__(self, idx):
        """Get item from dataset."""
        return self.X[idx], self.objectives[idx]


class JaxDataLoader:
    """Dataloader for DC3 dataset implemented in JAX."""

    def __init__(
        self,
        filepath,
        use_convex,
        batch_size: int,
        shuffle: bool = True,
        rng_key: Optional[jax.Array] = None,
    ):
        """Initialize JaxDataLoader."""
        self.dataset = DC3Dataset(filepath, use_convex)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
        # Batch indices for the current epoch
        if self.shuffle:
            self._perm = self._get_perm()
        else:
            self._perm = jnp.arange(len(self.dataset))

    def __len__(self):
        """Length of dataset."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """Iterate over the dataset."""
        for start in range(0, len(self.dataset), self.batch_size):
            batch_idx = self._perm[start : start + self.batch_size]
            yield self.dataset[batch_idx]

        if self.shuffle:
            self._perm = self._get_perm()

    def _advance_rng(self):
        self.rng_key, self._last_key = jax.random.split(self.rng_key)

    def _get_perm(self):
        self._advance_rng()
        perm = jax.random.permutation(self._last_key, len(self.dataset))
        return perm


def dc3_dataloader(
    filepath,
    use_convex,
    batch_size=512,
    shuffle=True,
):
    """Dataset loader for training, or validation, or test."""
    dataset = DC3Dataset(filepath, use_convex)

    def collate_fn(batch):
        X, obj = zip(*batch)
        return jnp.array(X), jnp.array(obj)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )

    return loader


def load_data(
    use_DC3_dataset,
    use_convex,
    problem_seed,
    problem_var,
    problem_nineq,
    problem_neq,
    problem_examples,
    rng_key,
    batch_size=2048,
    use_jax_loader=True,
):
    """Load problem data."""
    if not use_DC3_dataset:
        # Choose problem parameters
        if use_convex:
            filename = (
                f"SimpleQP_seed{problem_seed}_var{problem_var}_ineq{problem_nineq}"
                f"_eq{problem_neq}_examples{problem_examples}.npz"
            )
        else:
            raise NotImplementedError()
        dataset_path = os.path.join(os.path.dirname(__file__), "datasets", filename)

        QPDataset = SimpleQPDataset(dataset_path)
        train_loader, valid_loader, test_loader = create_dataloaders(
            dataset_path, batch_size=batch_size, val_split=0.1, test_split=0.1
        )
        Q, p, A, G, h = QPDataset.const
        p = p[0, :, :]
        X = QPDataset.X
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
        dataset_path_train = os.path.join(
            os.path.dirname(__file__), "datasets", filename_train
        )
        filename_valid = filename + "valid.npz"
        dataset_path_valid = os.path.join(
            os.path.dirname(__file__), "datasets", filename_valid
        )
        filename_test = filename + "test.npz"
        dataset_path_test = os.path.join(
            os.path.dirname(__file__), "datasets", filename_test
        )
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
        Q, p, A, G, h = train_loader.dataset.const
        p = p[0, :, :]
        X = train_loader.dataset.X

    return (filename, Q, p, A, G, h, X, train_loader, valid_loader, test_loader)
