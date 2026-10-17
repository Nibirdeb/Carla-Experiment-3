from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import RGB_DIR, LANE_MASK_DIR, ROAD_MASK_DIR


class CarlaMultiTaskDataset(Dataset):
    """
    CARLA multi-task dataset.

    Each sample contains:
        - RGB image
        - Lane mask
        - Road mask
    """

    def __init__(self):
        self.rgb_dir = Path(RGB_DIR)
        self.lane_dir = Path(LANE_MASK_DIR)
        self.road_dir = Path(ROAD_MASK_DIR)

        # Get all RGB images
        self.image_files = sorted(
            self.rgb_dir.glob("*.png")
        )

        if len(self.image_files) == 0:
            raise RuntimeError(
                f"No RGB images found in {self.rgb_dir}"
            )

        # Verify that corresponding masks exist
        valid_files = []

        for image_path in self.image_files:
            filename = image_path.name

            lane_path = self.lane_dir / filename
            road_path = self.road_dir / filename

            if lane_path.exists() and road_path.exists():
                valid_files.append(image_path)
            else:
                print(
                    f"Warning: Missing mask for {filename}"
                )

        self.image_files = valid_files

        print(f"Dataset samples: {len(self.image_files)}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        # ----------------------------------------------------
        # File paths
        # ----------------------------------------------------

        image_path = self.image_files[index]

        filename = image_path.name

        lane_path = self.lane_dir / filename
        road_path = self.road_dir / filename

        # ----------------------------------------------------
        # Load RGB image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        # OpenCV loads BGR, convert to RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Convert from uint8 [0, 255]
        # to float32 [0, 1]
        image = image.astype(np.float32) / 255.0

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # Load lane mask
        # ----------------------------------------------------

        lane_mask = cv2.imread(
            str(lane_path),
            cv2.IMREAD_GRAYSCALE
        )

        if lane_mask is None:
            raise RuntimeError(
                f"Could not read lane mask: {lane_path}"
            )

        # Convert binary mask to 0/1
        lane_mask = (lane_mask > 0).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Load road mask
        # ----------------------------------------------------

        road_mask = cv2.imread(
            str(road_path),
            cv2.IMREAD_GRAYSCALE
        )

        if road_mask is None:
            raise RuntimeError(
                f"Could not read road mask: {road_path}"
            )

        # Convert binary mask to 0/1
        road_mask = (road_mask > 0).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Convert to PyTorch tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            image.copy()
        ).float()

        lane_mask = torch.from_numpy(
            lane_mask.copy()
        ).float()

        road_mask = torch.from_numpy(
            road_mask.copy()
        ).float()

        # Add channel dimension to masks
        # H x W -> 1 x H x W
        lane_mask = lane_mask.unsqueeze(0)
        road_mask = road_mask.unsqueeze(0)

        return {
            "image": image,
            "lane_mask": lane_mask,
            "road_mask": road_mask,
        }


if __name__ == "__main__":

    dataset = CarlaMultiTaskDataset()

    print("\nDataset test")
    print("-" * 50)

    sample = dataset[0]

    print("Number of samples:", len(dataset))
    print("Image shape:", sample["image"].shape)
    print("Lane mask shape:", sample["lane_mask"].shape)
    print("Road mask shape:", sample["road_mask"].shape)

    print(
        "Image range:",
        sample["image"].min().item(),
        "to",
        sample["image"].max().item()
    )

    print(
        "Lane mask values:",
        torch.unique(sample["lane_mask"]).tolist()
    )

    print(
        "Road mask values:",
        torch.unique(sample["road_mask"]).tolist()
    )