
from pathlib import Path
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

RGB_DIR = PROJECT_DIR / "data" / "rgb"
LANE_DIR = PROJECT_DIR / "data" / "lane_masks"
ROAD_DIR = PROJECT_DIR / "data" / "road_masks"

OUTPUT_DIR = PROJECT_DIR / "outputs" / "visual_label_audit"

IMAGE_WIDTH = 1248
IMAGE_HEIGHT = 384

RANDOM_SEED = 42
NUMBER_OF_RANDOM_SAMPLES = 20

# Previously observed images containing crosswalks.
SPECIAL_SAMPLES = ["003022", "004383"]


def find_image(folder, stem):
    """Find an image by filename, allowing common image extensions."""
    for extension in [".png", ".jpg", ".jpeg", ".bmp"]:
        path = folder / f"{stem}{extension}"
        if path.exists():
            return path
    return None


def load_rgb(path):
    """Load an RGB image without modifying the original file."""
    with Image.open(path) as image:
        image.load()
        return np.array(image.convert("RGB"))


def load_mask(path):
    """Load a mask as a grayscale array and threshold it to binary."""
    with Image.open(path) as image:
        image.load()
        mask = np.array(image.convert("L"))

    return mask, mask > 0


def create_audit_figure(stem, output_path):
    """Save RGB, masks, and overlays for one sample."""

    rgb_path = find_image(RGB_DIR, stem)
    lane_path = find_image(LANE_DIR, stem)
    road_path = find_image(ROAD_DIR, stem)

    missing = [
        name
        for name, path in [
            ("RGB", rgb_path),
            ("lane mask", lane_path),
            ("road mask", road_path),
        ]
        if path is None
    ]

    if missing:
        print(f"SKIPPED {stem}: missing {', '.join(missing)}")
        return False

    try:
        rgb = load_rgb(rgb_path)
        lane_gray, lane_binary = load_mask(lane_path)
        road_gray, road_binary = load_mask(road_path)

        expected_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)

        if (
            rgb.shape[:2] != expected_shape
            or lane_binary.shape != expected_shape
            or road_binary.shape != expected_shape
        ):
            print(f"WARNING {stem}: unexpected image or mask dimensions")

        # Make overlays for visual inspection.
        lane_overlay = rgb.copy()
        lane_overlay[lane_binary] = (
            0.5 * rgb[lane_binary]
            + 0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)

        road_overlay = rgb.copy()
        road_overlay[road_binary] = (
            0.5 * rgb[road_binary]
            + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)

        fig, axes = plt.subplots(2, 3, figsize=(18, 7))

        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title("RGB image")

        axes[0, 1].imshow(lane_gray, cmap="gray", vmin=0, vmax=255)
        axes[0, 1].set_title("Lane mask")

        axes[0, 2].imshow(road_gray, cmap="gray", vmin=0, vmax=255)
        axes[0, 2].set_title("Road mask")

        axes[1, 0].imshow(lane_overlay)
        axes[1, 0].set_title("Lane overlay (red)")

        axes[1, 1].imshow(road_overlay)
        axes[1, 1].set_title("Road overlay (green)")

        # White indicates pixels positive in both masks.
        combined = np.zeros((*expected_shape, 3), dtype=np.uint8)
        combined[lane_binary] = [255, 0, 0]
        combined[road_binary] = [0, 255, 0]
        combined[lane_binary & road_binary] = [255, 255, 0]

        axes[1, 2].imshow(combined)
        axes[1, 2].set_title("Mask comparison: red=lane, green=road")

        for axis in axes.flat:
            axis.axis("off")

        lane_percentage = 100 * lane_binary.mean()
        road_percentage = 100 * road_binary.mean()

        fig.suptitle(
            f"Sample {stem} | "
            f"Lane foreground: {lane_percentage:.3f}% | "
            f"Road foreground: {road_percentage:.3f}%",
            fontsize=14,
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_path.name}")
        return True

    except Exception as error:
        print(f"ERROR processing {stem}: {error}")
        plt.close("all")
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(
        path for path in RGB_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )

    available_stems = {path.stem for path in rgb_files}

    rng = random.Random(RANDOM_SEED)
    random_candidates = sorted(
        available_stems - set(SPECIAL_SAMPLES)
    )

    sample_count = min(
        NUMBER_OF_RANDOM_SAMPLES,
        len(random_candidates),
    )

    random_samples = rng.sample(random_candidates, sample_count)

    # Special samples first, followed by reproducible random samples.
    selected_samples = list(dict.fromkeys(
        SPECIAL_SAMPLES + random_samples
    ))

    successful = 0

    print("=" * 60)
    print("VISUAL LABEL-QUALITY AUDIT")
    print("=" * 60)
    print(f"RGB images found: {len(rgb_files)}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Random samples requested: {NUMBER_OF_RANDOM_SAMPLES}")
    print(f"Special samples: {SPECIAL_SAMPLES}")
    print(f"Total samples selected: {len(selected_samples)}")
    print(f"Output folder: {OUTPUT_DIR}")
    print()

    for index, stem in enumerate(selected_samples, start=1):
        print(f"[{index}/{len(selected_samples)}] Inspecting {stem}")

        output_path = OUTPUT_DIR / f"audit_{stem}.png"

        if create_audit_figure(stem, output_path):
            successful += 1

    # Save the exact list so the audit can be reproduced.
    manifest_path = OUTPUT_DIR / "sample_manifest.txt"
    manifest_path.write_text(
        "\n".join(selected_samples) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print("AUDIT COMPLETE")
    print(f"Samples selected: {len(selected_samples)}")
    print(f"Figures saved: {successful}")
    print(f"Manifest saved: {manifest_path}")
    print(f"Evidence folder: {OUTPUT_DIR}")
    print()
    print(
        "IMPORTANT: Visual inspection is needed to determine "
        "whether the labels are semantically correct."
    )


if __name__ == "__main__":
    main()
