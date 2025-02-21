"""Loading functionality for simple QP benchmark."""

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

    def __init__(self, filepath):
        """Initialize dataset."""
        data = jnp.load(filepath)
        # Parameter values for each instance
        self.X = data["X"]
        # Constant problem ingredients
        self.const = (data["Q"], data["p"], data["A"], data["G"], data["h"])
        # Problem solutions
        self.Ystar = data["Ystar"]

        # Compute objectives
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


def dc3_dataloader(
    filepath,
    batch_size=512,
    shuffle=True,
):
    """Dataset loader for training, or validation, or test."""
    dataset = DC3Dataset(filepath)

    def collate_fn(batch):
        X, obj = zip(*batch)
        return jnp.array(X), jnp.array(obj)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )

    return loader
