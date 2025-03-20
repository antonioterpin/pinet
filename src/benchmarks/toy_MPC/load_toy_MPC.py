"""Loading functionality for toy MPC benchmark."""

import jax.numpy as jnp
from torch.utils.data import DataLoader, Dataset, random_split


# Load Instance Dataset
class ToyMPCDataset(Dataset):
    """Dataset for toy MPC benchmark."""

    def __init__(self, filepath):
        """Initialize dataset."""
        data = jnp.load(filepath)
        # Parameter values for each instance
        self.x0sets = data["x0sets"]
        # Constant problem ingredients
        self.const = (
            data["As"],
            data["lbxs"],
            data["ubxs"],
            data["lbus"],
            data["ubus"],
            data["xhat"],
            data["alpha"],
            data["T"],
            data["base_dim"],
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
    filepath, batch_size=2048, val_split=0.1, test_split=0.1, shuffle=True
):
    """Dataset loaders for training, validation and test."""
    dataset = ToyMPCDataset(filepath)
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
