import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.contiguous().view(
            predictions.size(0), -1
        )

        targets = targets.contiguous().view(
            targets.size(0), -1
        )

        intersection = (
            predictions * targets
        ).sum(dim=1)

        dice = (
            (2.0 * intersection + self.smooth)
            /
            (
                predictions.sum(dim=1)
                + targets.sum(dim=1)
                + self.smooth
            )
        )

        return (1.0 - dice).mean()


class BinarySegmentationLoss(nn.Module):
    """
    Combination of Binary Cross Entropy
    and Dice Loss.
    """

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5
    ):

        super().__init__()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, predictions, targets):

        bce_loss = self.bce(
            predictions,
            targets
        )

        dice_loss = self.dice(
            predictions,
            targets
        )

        total_loss = (
            self.bce_weight * bce_loss
            +
            self.dice_weight * dice_loss
        )

        return total_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining lane detection
    and road segmentation losses.

    Total Loss =
        lane_weight * lane_loss
        +
        road_weight * road_loss
    """

    def __init__(
        self,
        lane_weight=1.0,
        road_weight=1.0
    ):

        super().__init__()

        self.lane_weight = lane_weight
        self.road_weight = road_weight

        self.segmentation_loss = (
            BinarySegmentationLoss()
        )

    def forward(
        self,
        outputs,
        lane_targets,
        road_targets
    ):

        lane_loss = self.segmentation_loss(
            outputs["lane"],
            lane_targets
        )

        road_loss = self.segmentation_loss(
            outputs["road"],
            road_targets
        )

        total_loss = (
            self.lane_weight * lane_loss
            +
            self.road_weight * road_loss
        )

        return {
            "total": total_loss,
            "lane": lane_loss,
            "road": road_loss
        }


if __name__ == "__main__":

    print("=" * 60)
    print("Multi-Task Loss Test")
    print("=" * 60)

    batch_size = 2
    height = 384
    width = 1248

    lane_prediction = torch.randn(
        batch_size,
        1,
        height,
        width
    )

    road_prediction = torch.randn(
        batch_size,
        1,
        height,
        width
    )

    lane_target = torch.randint(
        0,
        2,
        (
            batch_size,
            1,
            height,
            width
        )
    ).float()

    road_target = torch.randint(
        0,
        2,
        (
            batch_size,
            1,
            height,
            width
        )
    ).float()

    outputs = {
        "lane": lane_prediction,
        "road": road_prediction
    }

    criterion = MultiTaskLoss()

    losses = criterion(
        outputs,
        lane_target,
        road_target
    )

    print("Lane loss:", losses["lane"].item())
    print("Road loss:", losses["road"].item())
    print("Total loss:", losses["total"].item())

    print("=" * 60)
    print("Loss test completed successfully.")
    print("=" * 60)