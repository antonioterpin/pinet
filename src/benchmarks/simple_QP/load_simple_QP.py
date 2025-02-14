"""Loading functionality for simple QP benchmark."""

import jax.numpy as jnp
from torch.utils.data import DataLoader, Dataset, random_split

from benchmarks.simple_QP.generate_simple_QP import optimal_objectives


# Load Instance Dataset
class SimpleQPDataset(Dataset):
    """Dataset for simple QP benchmark."""

    def __init__(self, filepath):
        """Initialize dataset."""
        data = jnp.load(filepath)
        self.objectives = optimal_objectives(filepath)
        # Parameter values for each instance
        self.X = data["X"]
        # Constant problem ingredients
        self.const = (data["Q"], data["p"], data["A"], data["G"], data["h"])

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
