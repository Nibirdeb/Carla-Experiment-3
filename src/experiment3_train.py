
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

# Make existing project modules importable when run from project root.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from model import MultiTaskUNet
from loss import MultiTaskLoss


# -------------------- Configuration --------------------
MANIFEST = BASE_DIR / "experiment3_controlled" / "split_manifest.csv"
OUTPUT_DIR = BASE_DIR / "experiment3_controlled"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
SEED = 42

EXPECTED_COUNTS = {"train": 3500, "val": 750, "test": 750}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ManifestDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def read_rgb(path):
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    @staticmethod
    def read_mask(path):
        image = Image.open(path).convert("L")
        array = np.asarray(image, dtype=np.uint8)
        array = (array > 0).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0).contiguous()

    def __getitem__(self, index):
        row = self.rows[index]
        rgb = self.read_rgb(row["rgb_path"])
        lane = self.read_mask(row["lane_mask_path"])
        road = self.read_mask(row["road_mask_path"])

        if rgb.shape != (3, 384, 1248):
            raise ValueError(
                f"Unexpected RGB shape for {row['filename']}: {tuple(rgb.shape)}"
            )
        if lane.shape != (1, 384, 1248) or road.shape != (1, 384, 1248):
            raise ValueError(f"Unexpected mask shape for {row['filename']}")

        return rgb, lane, road


def load_and_validate_manifest():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {
            "split", "sample_id", "filename",
            "rgb_path", "lane_mask_path", "road_mask_path"
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Manifest must contain columns: {sorted(required)}"
            )
        rows = list(reader)

    split_rows = {name: [] for name in EXPECTED_COUNTS}
    seen_ids = set()

    for row in rows:
        split = row["split"].strip()
        if split not in split_rows:
            raise ValueError(f"Unknown split name: {split}")

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

        split_rows[split].append(row)

    for split, expected in EXPECTED_COUNTS.items():
        actual = len(split_rows[split])
        if actual != expected:
            raise ValueError(
                f"{split} split has {actual} rows; expected {expected}"
            )

    train_ids = {int(r["sample_id"]) for r in split_rows["train"]}
    val_ids = {int(r["sample_id"]) for r in split_rows["val"]}
    test_ids = {int(r["sample_id"]) for r in split_rows["test"]}

    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Split IDs overlap.")
    if train_ids != set(range(1, 3501)):
        raise ValueError("Training IDs are not exactly 1 through 3500.")
    if val_ids != set(range(3501, 4251)):
        raise ValueError("Validation IDs are not exactly 3501 through 4250.")
    if test_ids != set(range(4251, 5001)):
        raise ValueError("Test IDs are not exactly 4251 through 5000.")

    print("Manifest verified.")
    print("Train: 3,500 | Validation: 750 | Test: 750")
    print("Test rows will not be loaded by this training script.")

    return split_rows["train"], split_rows["val"]


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    lane_loss_sum = 0.0
    road_loss_sum = 0.0
    sample_count = 0

    for rgb, lane, road in loader:
        rgb = rgb.to(device, non_blocking=True)
        lane = lane.to(device, non_blocking=True)
        road = road.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(rgb)
            losses = criterion(outputs, lane, road)
            loss = losses["total"]

            if training:
                loss.backward()
                optimizer.step()

        batch_size = rgb.size(0)
        total_loss += loss.item() * batch_size
        lane_loss_sum += losses["lane"].item() * batch_size
        road_loss_sum += losses["road"].item() * batch_size
        sample_count += batch_size

    return {
        "loss": total_loss / sample_count,
        "lane_loss": lane_loss_sum / sample_count,
        "road_loss": road_loss_sum / sample_count,
    }


def main():
    set_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Avoid accidentally replacing results from a previous controlled run.
    protected_outputs = [
        CHECKPOINT_DIR / "best_model.pth",
        CHECKPOINT_DIR / "latest.pth",
        LOG_DIR / "training_history.csv",
    ]
    existing = [p for p in protected_outputs if p.exists()]
    if existing:
        raise FileExistsError(
            "Controlled-run outputs already exist; nothing was overwritten:\n"
            + "\n".join(str(p) for p in existing)
            + "\nMove or back up these controlled-run files before retrying."
        )

    train_rows, val_rows = load_and_validate_manifest()

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = MultiTaskUNet().to(device)
    criterion = MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history_path = LOG_DIR / "training_history.csv"
    best_val_loss = float("inf")

    with history_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "epoch", "train_loss", "train_lane_loss", "train_road_loss",
            "val_loss", "val_lane_loss", "val_road_loss", "learning_rate"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, EPOCHS + 1):
            train_metrics = run_epoch(
                model, train_loader, criterion, device, optimizer
            )
            val_metrics = run_epoch(
                model, val_loader, criterion, device
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

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "seed": SEED,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "manifest": str(MANIFEST),
                "best_validation_loss_so_far": min(
                    best_val_loss, val_metrics["loss"]
                ),
            }

            torch.save(checkpoint, CHECKPOINT_DIR / "latest.pth")

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                checkpoint["best_validation_loss_so_far"] = best_val_loss
                torch.save(checkpoint, CHECKPOINT_DIR / "best_model.pth")

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"train={train_metrics['loss']:.5f} | "
                f"val={val_metrics['loss']:.5f} | "
                f"lane_val={val_metrics['lane_loss']:.5f} | "
                f"road_val={val_metrics['road_loss']:.5f}"
            )

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best checkpoint: {CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"Latest checkpoint: {CHECKPOINT_DIR / 'latest.pth'}")
    print(f"History: {history_path}")
    print("The test split has not been evaluated.")


if __name__ == "__main__":
    main()

