
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model import MultiTaskUNet


ROOT = Path(__file__).resolve().parents[1]
CONTROLLED = ROOT / "experiment3_controlled"
MANIFEST = CONTROLLED / "split_manifest.csv"
CHECKPOINT = CONTROLLED / "checkpoints" / "best_model.pth"
OUTPUT_DIR = CONTROLLED / "evaluation"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
THRESHOLD = 0.5


class TestDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        rgb = np.asarray(
            Image.open(row["rgb_path"]).convert("RGB"),
            dtype=np.float32,
        ) / 255.0

        lane = np.asarray(
            Image.open(row["lane_mask_path"]).convert("L"),
            dtype=np.float32,
        )
        road = np.asarray(
            Image.open(row["road_mask_path"]).convert("L"),
            dtype=np.float32,
        )

        # Match the usual RGB tensor layout: C x H x W.
        rgb = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        lane = torch.from_numpy((lane > 127).astype(np.float32)[None, ...])
        road = torch.from_numpy((road > 127).astype(np.float32)[None, ...])

        return rgb, lane, road, row["filename"]


def calculate_metrics(prediction, target):
    pred = prediction.astype(bool)
    truth = target.astype(bool)

    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())

    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    dice = (
        2 * tp / (2 * tp + fp + fn)
        if 2 * tp + fp + fn else 1.0
    )
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST}")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")

    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as file:
        all_rows = list(csv.DictReader(file))

    test_rows = [row for row in all_rows if row["split"] == "test"]
    if len(test_rows) != 750:
        raise ValueError(f"Expected 750 test samples, found {len(test_rows)}")

    # Check only test paths; do not load training or validation images.
    for row in test_rows:
        for key in ("rgb_path", "lane_mask_path", "road_mask_path"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(f"Missing test file: {row[key]}")

    dataset = TestDataset(test_rows)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model = MultiTaskUNet().to(DEVICE)
    state = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    if state is None:
        raise KeyError(
            "Could not find model weights in checkpoint. "
            f"Checkpoint keys: {list(checkpoint.keys())}"
        )
    model.load_state_dict(state)
    model.eval()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Evaluating test samples only: {len(test_rows)}")

    per_image = []

    with torch.no_grad():
        for batch_index, (images, lane_masks, road_masks, filenames) in enumerate(loader):
            images = images.to(DEVICE)
            lane_masks = lane_masks.numpy() > 0.5
            road_masks = road_masks.numpy() > 0.5

            outputs = model(images)
            lane_probs = torch.sigmoid(outputs["lane"]).cpu().numpy()
            road_probs = torch.sigmoid(outputs["road"]).cpu().numpy()

            lane_preds = lane_probs >= THRESHOLD
            road_preds = road_probs >= THRESHOLD

            for i, filename in enumerate(filenames):
                lane_metrics = calculate_metrics(lane_preds[i], lane_masks[i])
                road_metrics = calculate_metrics(road_preds[i], road_masks[i])

                row = {"filename": filename}
                for task, metrics in (
                    ("lane", lane_metrics),
                    ("road", road_metrics),
                ):
                    for metric, value in metrics.items():
                        row[f"{task}_{metric}"] = value
                per_image.append(row)

            done = min((batch_index + 1) * BATCH_SIZE, len(test_rows))
            if done % 100 == 0 or done == len(test_rows):
                print(f"Processed {done}/{len(test_rows)} test images")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_image_path = OUTPUT_DIR / "test_metrics_per_image.csv"
    metric_names = [
        f"{task}_{metric}"
        for task in ("lane", "road")
        for metric in ("iou", "dice", "precision", "recall", "tp", "fp", "fn")
    ]

    with per_image_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["filename"] + metric_names)
        writer.writeheader()
        writer.writerows(per_image)

    summary_path = OUTPUT_DIR / "test_metrics_summary.csv"
    summary = []
    for task in ("lane", "road"):
        for metric in ("iou", "dice", "precision", "recall"):
            values = [float(row[f"{task}_{metric}"]) for row in per_image]
            summary.append({
                "task": task,
                "metric": metric,
                "mean_per_image": float(np.mean(values)),
                "std_per_image": float(np.std(values)),
            })

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["task", "metric", "mean_per_image", "std_per_image"],
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\nTest-set summary (mean per image)")
    for row in summary:
        print(
            f"{row['task']:>4} | {row['metric']:<9} | "
            f"{row['mean_per_image']:.6f} "
            f"(std {row['std_per_image']:.6f})"
        )

    print(f"\nPer-image metrics saved to: {per_image_path}")
    print(f"Summary metrics saved to: {summary_path}")
    print("Test evaluation complete.")


if __name__ == "__main__":
    main()
