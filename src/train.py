import csv
from pathlib import Path

import torch
from torch.optim import Adam
from tqdm import tqdm

from config import (
    DEVICE,
    LEARNING_RATE,
    NUM_EPOCHS,
    CHECKPOINT_DIR,
    LOG_DIR,
)
from dataloader import create_dataloaders
from model import MultiTaskUNet
from loss import MultiTaskLoss


# ============================================================
# Metrics
# ============================================================

def calculate_binary_metrics(predictions, targets, threshold=0.5):
    """
    Calculate IoU, Dice, Precision and Recall
    for binary segmentation.
    """

    probabilities = torch.sigmoid(predictions)

    predicted = (probabilities >= threshold).float()
    targets = (targets >= threshold).float()

    predicted = predicted.view(predicted.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    true_positive = (predicted * targets).sum(dim=1)

    false_positive = (
        predicted * (1.0 - targets)
    ).sum(dim=1)

    false_negative = (
        (1.0 - predicted) * targets
    ).sum(dim=1)

    intersection = true_positive

    union = (
        true_positive
        + false_positive
        + false_negative
    )

    # IoU
    iou = (
        intersection + 1e-7
    ) / (
        union + 1e-7
    )

    # Dice
    dice = (
        2.0 * true_positive + 1e-7
    ) / (
        2.0 * true_positive
        + false_positive
        + false_negative
        + 1e-7
    )

    # Precision
    precision = (
        true_positive + 1e-7
    ) / (
        true_positive
        + false_positive
        + 1e-7
    )

    # Recall
    recall = (
        true_positive + 1e-7
    ) / (
        true_positive
        + false_negative
        + 1e-7
    )

    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    epoch,
):
    model.train()

    total_loss = 0.0
    total_lane_loss = 0.0
    total_road_loss = 0.0

    progress_bar = tqdm(
        loader,
        desc=f"Epoch {epoch} Training",
        leave=True,
    )

    for batch in progress_bar:

        images = batch["image"].to(
            DEVICE,
            non_blocking=True,
        )

        lane_targets = batch["lane_mask"].to(
            DEVICE,
            non_blocking=True,
        )

        road_targets = batch["road_mask"].to(
            DEVICE,
            non_blocking=True,
        )

        # Clear gradients
        optimizer.zero_grad(
            set_to_none=True
        )

        # Forward pass
        outputs = model(images)

        # Calculate multi-task loss
        losses = criterion(
            outputs,
            lane_targets,
            road_targets,
        )

        # Backpropagation
        losses["total"].backward()

        # Update weights
        optimizer.step()

        # Accumulate losses
        total_loss += losses["total"].item()
        total_lane_loss += losses["lane"].item()
        total_road_loss += losses["road"].item()

        progress_bar.set_postfix(
            loss=f"{losses['total'].item():.4f}"
        )

    num_batches = len(loader)

    return {
        "total_loss": total_loss / num_batches,
        "lane_loss": total_lane_loss / num_batches,
        "road_loss": total_road_loss / num_batches,
    }


# ============================================================
# Validation
# ============================================================

def validate(
    model,
    loader,
    criterion,
):
    model.eval()

    total_loss = 0.0
    total_lane_loss = 0.0
    total_road_loss = 0.0

    lane_iou = 0.0
    lane_dice = 0.0
    lane_precision = 0.0
    lane_recall = 0.0

    road_iou = 0.0
    road_dice = 0.0
    road_precision = 0.0
    road_recall = 0.0

    with torch.no_grad():

        progress_bar = tqdm(
            loader,
            desc="Validation",
            leave=True,
        )

        for batch in progress_bar:

            images = batch["image"].to(
                DEVICE,
                non_blocking=True,
            )

            lane_targets = batch["lane_mask"].to(
                DEVICE,
                non_blocking=True,
            )

            road_targets = batch["road_mask"].to(
                DEVICE,
                non_blocking=True,
            )

            # Forward pass
            outputs = model(images)

            # Loss
            losses = criterion(
                outputs,
                lane_targets,
                road_targets,
            )

            total_loss += losses["total"].item()
            total_lane_loss += losses["lane"].item()
            total_road_loss += losses["road"].item()

            # Lane metrics
            lane_metrics = calculate_binary_metrics(
                outputs["lane"],
                lane_targets,
            )

            lane_iou += lane_metrics["iou"]
            lane_dice += lane_metrics["dice"]
            lane_precision += lane_metrics["precision"]
            lane_recall += lane_metrics["recall"]

            # Road metrics
            road_metrics = calculate_binary_metrics(
                outputs["road"],
                road_targets,
            )

            road_iou += road_metrics["iou"]
            road_dice += road_metrics["dice"]
            road_precision += road_metrics["precision"]
            road_recall += road_metrics["recall"]

    num_batches = len(loader)

    return {
        "total_loss": total_loss / num_batches,
        "lane_loss": total_lane_loss / num_batches,
        "road_loss": total_road_loss / num_batches,

        "lane_iou": lane_iou / num_batches,
        "lane_dice": lane_dice / num_batches,
        "lane_precision": lane_precision / num_batches,
        "lane_recall": lane_recall / num_batches,

        "road_iou": road_iou / num_batches,
        "road_dice": road_dice / num_batches,
        "road_precision": road_precision / num_batches,
        "road_recall": road_recall / num_batches,
    }


# ============================================================
# Main Training Function
# ============================================================

def main():

    print("=" * 70)
    print("EXPERIMENT 3 - MULTI-TASK LEARNING")
    print("=" * 70)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader, val_loader = create_dataloaders()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MultiTaskUNet().to(DEVICE)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Trainable parameters: {total_parameters:,}"
    )

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = MultiTaskLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = []

    best_val_loss = float("inf")

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):

        print()
        print("=" * 70)
        print(
            f"Epoch {epoch}/{NUM_EPOCHS}"
        )
        print("=" * 70)

        # Training
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
        )

        # Validation
        val_metrics = validate(
            model,
            val_loader,
            criterion,
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print()
        print(
            f"Training Loss:   {train_metrics['total_loss']:.6f}"
        )

        print(
            f"  Lane Loss:     {train_metrics['lane_loss']:.6f}"
        )

        print(
            f"  Road Loss:     {train_metrics['road_loss']:.6f}"
        )

        print()
        print(
            f"Validation Loss: {val_metrics['total_loss']:.6f}"
        )

        print(
            f"  Lane Loss:     {val_metrics['lane_loss']:.6f}"
        )

        print(
            f"  Road Loss:     {val_metrics['road_loss']:.6f}"
        )

        print()
        print(
            f"Lane IoU:        {val_metrics['lane_iou']:.6f}"
        )

        print(
            f"Lane Dice:       {val_metrics['lane_dice']:.6f}"
        )

        print(
            f"Lane Precision:  {val_metrics['lane_precision']:.6f}"
        )

        print(
            f"Lane Recall:     {val_metrics['lane_recall']:.6f}"
        )

        print()
        print(
            f"Road IoU:        {val_metrics['road_iou']:.6f}"
        )

        print(
            f"Road Dice:       {val_metrics['road_dice']:.6f}"
        )

        print(
            f"Road Precision:  {val_metrics['road_precision']:.6f}"
        )

        print(
            f"Road Recall:     {val_metrics['road_recall']:.6f}"
        )

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        epoch_results = {
            "epoch": epoch,

            "train_total_loss":
                train_metrics["total_loss"],

            "train_lane_loss":
                train_metrics["lane_loss"],

            "train_road_loss":
                train_metrics["road_loss"],

            "val_total_loss":
                val_metrics["total_loss"],

            "val_lane_loss":
                val_metrics["lane_loss"],

            "val_road_loss":
                val_metrics["road_loss"],

            "lane_iou":
                val_metrics["lane_iou"],

            "lane_dice":
                val_metrics["lane_dice"],

            "lane_precision":
                val_metrics["lane_precision"],

            "lane_recall":
                val_metrics["lane_recall"],

            "road_iou":
                val_metrics["road_iou"],

            "road_dice":
                val_metrics["road_dice"],

            "road_precision":
                val_metrics["road_precision"],

            "road_recall":
                val_metrics["road_recall"],
        }

        history.append(epoch_results)

        # ----------------------------------------------------
        # Save latest checkpoint
        # ----------------------------------------------------

        latest_checkpoint = (
            CHECKPOINT_DIR / "latest.pth"
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "val_loss":
                    val_metrics["total_loss"],
            },
            latest_checkpoint,
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if (
            val_metrics["total_loss"]
            < best_val_loss
        ):

            best_val_loss = (
                val_metrics["total_loss"]
            )

            best_checkpoint = (
                CHECKPOINT_DIR / "best_model.pth"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "val_loss":
                        best_val_loss,
                },
                best_checkpoint,
            )

            print()
            print(
                "✓ Best model checkpoint saved."
            )

    # --------------------------------------------------------
    # Save CSV history
    # --------------------------------------------------------

    history_file = (
        LOG_DIR / "training_history.csv"
    )

    fieldnames = list(
        history[0].keys()
    )

    with open(
        history_file,
        "w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print(
        f"Best validation loss: {best_val_loss:.6f}"
    )

    print(
        f"Best checkpoint: "
        f"{CHECKPOINT_DIR / 'best_model.pth'}"
    )

    print(
        f"Training history: "
        f"{history_file}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()