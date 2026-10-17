
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from config import DEVICE, CHECKPOINT_DIR, OUTPUT_DIR
from dataloader import create_dataloaders
from model import MultiTaskUNet


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate binary segmentation metrics."""

    predictions = torch.sigmoid(predictions)
    predictions = (predictions >= threshold).float()
    targets = (targets >= 0.5).float()

    predictions = predictions.reshape(predictions.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)

    tp = (predictions * targets).sum(dim=1)
    fp = (predictions * (1 - targets)).sum(dim=1)
    fn = ((1 - predictions) * targets).sum(dim=1)

    eps = 1e-7

    iou = (tp + eps) / (tp + fp + fn + eps)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# Visualization
# ============================================================

def save_prediction_figure(image, lane_gt, road_gt,
                           lane_pred, road_pred, save_path):
    """Save an RGB image and its ground-truth/predicted masks."""

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("RGB Image")

    axes[0, 1].imshow(lane_gt, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Lane Ground Truth")

    axes[0, 2].imshow(lane_pred, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Lane Prediction")

    axes[1, 0].imshow(image)
    axes[1, 0].set_title("RGB Image")

    axes[1, 1].imshow(road_gt, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Road Ground Truth")

    axes[1, 2].imshow(road_pred, cmap="gray", vmin=0, vmax=1)
    axes[1, 2].set_title("Road Prediction")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main Evaluation
# ============================================================

def main():

    print("=" * 65)
    print("EXPERIMENT 3 - BEST CHECKPOINT EVALUATION")
    print("=" * 65)
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Create output directories
    results_dir = OUTPUT_DIR / "evaluation"
    predictions_dir = results_dir / "predictions"

    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Recreate the seeded data split used for training.
    # random_split uses seed 42 in dataloader.py.
    _, val_loader = create_dataloaders()

    # Load model
    model = MultiTaskUNet().to(DEVICE)

    checkpoint_path = CHECKPOINT_DIR / "best_model.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Checkpoint:", checkpoint_path)
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Validation samples:", len(val_loader.dataset))

    metric_names = ["iou", "dice", "precision", "recall"]

    sums = {
        task: {name: 0.0 for name in metric_names}
        for task in ["lane", "road"]
    }

    sample_count = 0
    saved_figures = 0
    max_figures = 12

    # Evaluate
    with torch.inference_mode():

        for batch_index, batch in enumerate(val_loader):

            images = batch["image"].to(DEVICE)
            lane_targets = batch["lane_mask"].to(DEVICE)
            road_targets = batch["road_mask"].to(DEVICE)

            outputs = model(images)

            lane_metrics = calculate_metrics(
                outputs["lane"], lane_targets
            )
            road_metrics = calculate_metrics(
                outputs["road"], road_targets
            )

            batch_size = images.size(0)
            sample_count += batch_size

            for name in metric_names:
                sums["lane"][name] += (
                    lane_metrics[name].sum().item()
                )
                sums["road"][name] += (
                    road_metrics[name].sum().item()
                )

            # Save example predictions
            for i in range(batch_size):

                if saved_figures >= max_figures:
                    break

                image = (
                    images[i].cpu().numpy().transpose(1, 2, 0)
                )
                image = np.clip(image, 0, 1)

                lane_gt = lane_targets[i, 0].cpu().numpy()
                road_gt = road_targets[i, 0].cpu().numpy()

                lane_pred = (
                    torch.sigmoid(outputs["lane"][i, 0]) >= 0.5
                ).cpu().numpy().astype(np.uint8)

                road_pred = (
                    torch.sigmoid(outputs["road"][i, 0]) >= 0.5
                ).cpu().numpy().astype(np.uint8)

                save_path = (
                    predictions_dir
                    / f"validation_sample_{saved_figures + 1:02d}.png"
                )

                save_prediction_figure(
                    image,
                    lane_gt,
                    road_gt,
                    lane_pred,
                    road_pred,
                    save_path,
                )

                saved_figures += 1

            print(
                f"Evaluated {min((batch_index + 1) * batch_size, len(val_loader.dataset))}"
                f"/{len(val_loader.dataset)} samples",
                end="\r",
            )

    # Calculate final per-image averages
    results = {}

    for task in ["lane", "road"]:
        results[task] = {
            name: sums[task][name] / sample_count
            for name in metric_names
        }

    print("\n")
    print("=" * 65)
    print("FINAL VALIDATION METRICS")
    print("=" * 65)

    for task in ["lane", "road"]:
        print(f"\n{task.upper()}")

        for name in metric_names:
            print(f"{name.capitalize():12s}: {results[task][name]:.6f}")

    # Save CSV
    metrics_path = results_dir / "validation_metrics.csv"

    with open(metrics_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["task", "metric", "value"])

        for task in ["lane", "road"]:
            for name in metric_names:
                writer.writerow([
                    task,
                    name,
                    results[task][name],
                ])

    print("\nMetrics saved to:", metrics_path)
    print("Prediction figures saved to:", predictions_dir)
    print("Figures saved:", saved_figures)
    print("=" * 65)


if __name__ == "__main__":
    main()
