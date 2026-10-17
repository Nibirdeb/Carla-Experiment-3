import torch
from torch.utils.data import DataLoader, random_split

from dataset import CarlaMultiTaskDataset
from config import BATCH_SIZE, NUM_WORKERS


def create_dataloaders():

    # --------------------------------------------------------
    # Load complete dataset
    # --------------------------------------------------------

    dataset = CarlaMultiTaskDataset()

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Total samples:      {len(dataset)}")
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader


if __name__ == "__main__":

    train_loader, val_loader = create_dataloaders()

    print("\nDataLoader test")
    print("-" * 50)

    batch = next(iter(train_loader))

    print("Image batch shape:", batch["image"].shape)
    print("Lane batch shape:", batch["lane_mask"].shape)
    print("Road batch shape:", batch["road_mask"].shape)

    print("Number of training batches:", len(train_loader))
    print("Number of validation batches:", len(val_loader))