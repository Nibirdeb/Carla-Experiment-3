
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------
# Project configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from model import MultiTaskUNet
from loss import MultiTaskLoss

MANIFEST = BASE_DIR / "experiment3_controlled" / "split_manifest.csv"
OUTPUT_DIR = BASE_DIR / "experiment3_controlled"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

LATEST = CHECKPOINT_DIR / "latest.pth"
BEST = CHECKPOINT_DIR / "best_model.pth"
HISTORY = LOG_DIR / "training_history.csv"

BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
SEED = 42

EXPECTED_COUNTS = {"train": 3500, "val": 750, "test": 750}


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class ManifestDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        with Image.open(row["rgb_path"]) as image:
            rgb_array = (
                np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            )

        with Image.open(row["lane_mask_path"]) as image:
            lane_array = np.asarray(image.convert("L"), dtype=np.uint8)

        with Image.open(row["road_mask_path"]) as image:
            road_array = np.asarray(image.convert("L"), dtype=np.uint8)

        rgb = (
            torch.from_numpy(rgb_array)
            .permute(2, 0, 1)
            .contiguous()
        )

        lane = torch.from_numpy(
            (lane_array > 0).astype(np.float32)
        ).unsqueeze(0).contiguous()

        road = torch.from_numpy(
            (road_array > 0).astype(np.float32)
        ).unsqueeze(0).contiguous()

        if rgb.shape != (3, 384, 1248):
            raise ValueError(
                f"Unexpected RGB shape for {row['filename']}: "
                f"{tuple(rgb.shape)}"
            )

        if lane.shape != (1, 384, 1248):
            raise ValueError(f"Unexpected lane mask shape: {row['filename']}")

        if road.shape != (1, 384, 1248):
            raise ValueError(f"Unexpected road mask shape: {row['filename']}")

        return rgb, lane, road


# --------------------------------------------------
# Manifest validation
# --------------------------------------------------
def load_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required = {
            "split",
            "sample_id",
            "filename",
            "rgb_path",
            "lane_mask_path",
            "road_mask_path",
        }

        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Manifest must contain columns: {sorted(required)}"
            )

        rows = list(reader)

    splits = {"train": [], "val": [], "test": []}
    seen_ids = set()

    for row in rows:
        split = row["split"].strip()

        if split not in splits:
            raise ValueError(f"Unknown split: {split}")

        sample_id = int(row["sample_id"])

        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample ID: {sample_id}")

        seen_ids.add(sample_id)

        for key in ("rgb_path", "lane_mask_path", "road_mask_path"):
            path = Path(row[key])

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {key} for {row['filename']}: {path}"
                )

        splits[split].append(row)

    for split, expected_count in EXPECTED_COUNTS.items():
        actual_count = len(splits[split])

        if actual_count != expected_count:
            raise ValueError(
                f"{split} has {actual_count} rows; "
                f"expected {expected_count}"
            )

    expected_ids = {
        "train": set(range(1, 3501)),
        "val": set(range(3501, 4251)),
        "test": set(range(4251, 5001)),
    }

    for split, wanted_ids in expected_ids.items():
        actual_ids = {int(row["sample_id"]) for row in splits[split]}

        if actual_ids != wanted_ids:
            raise ValueError(f"Unexpected sample IDs in {split} split.")

    print("Manifest verified: 3500 train, 750 validation, 750 test.")
    print("Test split will not be loaded during training.")

    return splits["train"], splits["val"]


# --------------------------------------------------
# Training / validation epoch
# --------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    # Metric names used by the training history.
    totals = {
        "loss": 0.0,
        "lane_loss": 0.0,
        "road_loss": 0.0,
    }

    count = 0

    for rgb, lane, road in loader:
        rgb = rgb.to(device, non_blocking=True)
        lane = lane.to(device, non_blocking=True)
        road = road.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(rgb)
            losses = criterion(outputs, lane, road)

            # MultiTaskLoss returns "total", "lane", and "road".
            if training:
                losses["total"].backward()
                optimizer.step()

        batch_size = rgb.size(0)

        totals["loss"] += losses["total"].item() * batch_size
        totals["lane_loss"] += losses["lane"].item() * batch_size
        totals["road_loss"] += losses["road"].item() * batch_size

        count += batch_size

    if count == 0:
        raise RuntimeError("The data loader returned no samples.")

    return {
        key: value / count
        for key, value in totals.items()
    }


# --------------------------------------------------
# Main training routine: resume from checkpoint
# --------------------------------------------------
def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Never silently start from scratch or overwrite missing progress.
    if not LATEST.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint not found: {LATEST}"
        )

    if not BEST.is_file():
        raise FileNotFoundError(
            f"Best checkpoint not found: {BEST}"
        )

    if not HISTORY.is_file():
        raise FileNotFoundError(
            f"Training history not found: {HISTORY}"
        )

    train_rows, val_rows = load_manifest()

    train_loader = DataLoader(
        ManifestDataset(train_rows),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        ManifestDataset(val_rows),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = MultiTaskUNet().to(device)
    criterion = MultiTaskLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # Restore the latest saved training state.
    checkpoint = torch.load(
        LATEST,
        map_location=device,
        weights_only=False,
    )

    saved_epoch = int(checkpoint["epoch"])
    start_epoch = saved_epoch + 1

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing model weights.")

    if "optimizer_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing optimizer state.")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    best_val_loss = float(
        checkpoint["best_validation_loss_so_far"]
    )

    # Check history before appending anything.
    with HISTORY.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        existing_rows = list(csv.DictReader(file))

    if len(existing_rows) != saved_epoch:
        raise ValueError(
            f"History has {len(existing_rows)} rows, but the latest "
            f"checkpoint is at epoch {saved_epoch}. "
            "No training was started."
        )

    if existing_rows:
        history_epochs = [int(row["epoch"]) for row in existing_rows]

        if history_epochs != list(range(1, saved_epoch + 1)):
            raise ValueError(
                "Training history has missing or unordered epochs. "
                "No training was started."
            )

        csv_best_loss = min(
            float(row["val_loss"]) for row in existing_rows
        )

        if abs(csv_best_loss - best_val_loss) > 1e-6:
            raise ValueError(
                "Best validation loss does not match the saved history. "
                "No training was started."
            )

    if start_epoch > EPOCHS:
        print(f"Training already reached epoch {saved_epoch}.")
        print("No additional training is required.")
        return

    fieldnames = [
        "epoch",
        "train_loss",
        "train_lane_loss",
        "train_road_loss",
        "val_loss",
        "val_lane_loss",
        "val_road_loss",
        "learning_rate",
    ]

    print(
        f"Resuming from epoch {start_epoch}; "
        f"target is epoch {EPOCHS}."
    )
    print(f"Best validation loss so far: {best_val_loss:.6f}")

    # Append rather than replace the previous nine epochs.
    with HISTORY.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        for epoch in range(start_epoch, EPOCHS + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
            )

            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_lane_loss": train_metrics["lane_loss"],
                "train_road_loss": train_metrics["road_loss"],
                "val_loss": val_metrics["loss"],
                "val_lane_loss": val_metrics["lane_loss"],
                "val_road_loss": val_metrics["road_loss"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }

            writer.writerow(row)
            file.flush()

            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "manifest": str(MANIFEST),
                "best_validation_loss_so_far": best_val_loss,
            }

            # Save the latest state after each completed epoch.
            torch.save(checkpoint_data, LATEST)

            # Update the best checkpoint only on improvement.
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                checkpoint_data["best_validation_loss_so_far"] = (
                    best_val_loss
                )
                torch.save(checkpoint_data, BEST)

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"train={train_metrics['loss']:.5f} | "
                f"val={val_metrics['loss']:.5f} | "
                f"lane_val={val_metrics['lane_loss']:.5f} | "
                f"road_val={val_metrics['road_loss']:.5f}",
                flush=True,
            )

    print("\nResume training complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {BEST}")
    print(f"Latest checkpoint: {LATEST}")
    print(f"Training history: {HISTORY}")
    print("Test split has not been evaluated.")


if __name__ == "__main__":
    main()
