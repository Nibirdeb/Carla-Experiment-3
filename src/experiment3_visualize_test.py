
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import MultiTaskUNet


ROOT = Path(__file__).resolve().parents[1]
CONTROLLED = ROOT / "experiment3_controlled"
MANIFEST = CONTROLLED / "split_manifest.csv"
CHECKPOINT = CONTROLLED / "checkpoints" / "best_model.pth"
OUTPUT_DIR = CONTROLLED / "evaluation" / "test_predictions"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = 0.5

# Fixed, evenly spaced examples from the held-out test split.
NUM_EXAMPLES = 12


def load_rgb(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    return image, tensor


def load_mask(path):
    image = Image.open(path).convert("L")
    array = np.asarray(image)
    binary = array > 127
    return Image.fromarray((binary * 255).astype(np.uint8)), binary


def color_overlay(rgb_image, mask_image, color, alpha=110):
    rgb = rgb_image.convert("RGBA")
    mask = np.asarray(mask_image.convert("L")) > 127
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[mask] = [*color, alpha]
    return Image.alpha_composite(rgb, Image.fromarray(overlay, mode="RGBA")).convert("RGB")


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    test_rows = sorted(
        [row for row in rows if row["split"] == "test"],
        key=lambda row: row["filename"],
    )

    if len(test_rows) != 750:
        raise ValueError(f"Expected 750 test samples, found {len(test_rows)}")

    # Choose 12 evenly spaced test images, not training or validation images.
    indices = np.linspace(0, len(test_rows) - 1, NUM_EXAMPLES, dtype=int)
    selected_rows = [test_rows[i] for i in indices]

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model = MultiTaskUNet().to(DEVICE)
    state = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
    if state is None:
        raise KeyError(f"No model weights found. Checkpoint keys: {list(checkpoint.keys())}")

    model.load_state_dict(state)
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for number, row in enumerate(selected_rows, start=1):
        rgb, tensor = load_rgb(row["rgb_path"])
        lane_gt, _ = load_mask(row["lane_mask_path"])
        road_gt, _ = load_mask(row["road_mask_path"])

        with torch.no_grad():
            outputs = model(tensor.unsqueeze(0).to(DEVICE))
            lane_prob = torch.sigmoid(outputs["lane"])[0, 0].cpu().numpy()
            road_prob = torch.sigmoid(outputs["road"])[0, 0].cpu().numpy()

        lane_pred = Image.fromarray(
            ((lane_prob >= THRESHOLD) * 255).astype(np.uint8)
        )
        road_pred = Image.fromarray(
            ((road_prob >= THRESHOLD) * 255).astype(np.uint8)
        )

        # Green = ground truth; red = prediction.
        lane_gt_overlay = color_overlay(rgb, lane_gt, (0, 255, 0))
        lane_pred_overlay = color_overlay(rgb, lane_pred, (255, 0, 0))
        road_gt_overlay = color_overlay(rgb, road_gt, (0, 255, 0))
        road_pred_overlay = color_overlay(rgb, road_pred, (255, 0, 0))

        panels = [
            ("RGB input", rgb),
            ("Lane ground truth", lane_gt),
            ("Lane prediction", lane_pred),
            ("Lane GT overlay", lane_gt_overlay),
            ("Lane prediction overlay", lane_pred_overlay),
            ("Road ground truth", road_gt),
            ("Road prediction", road_pred),
            ("Road GT overlay", road_gt_overlay),
            ("Road prediction overlay", road_pred_overlay),
        ]

        # Create a labeled 3 x 3 comparison sheet.
        panel_width, panel_height = rgb.size
        label_height = 30
        sheet = Image.new(
            "RGB",
            (panel_width * 3, (panel_height + label_height) * 3),
            (35, 35, 35),
        )
        draw = ImageDraw.Draw(sheet)

        for i, (label, image) in enumerate(panels):
            col = i % 3
            row_index = i // 3
            x = col * panel_width
            y = row_index * (panel_height + label_height)
            draw.text((x + 8, y + 7), label, fill=(255, 255, 255))
            sheet.paste(image.convert("RGB"), (x, y + label_height))

        filename = Path(row["filename"]).stem
        output_path = OUTPUT_DIR / f"{filename}_comparison.png"
        sheet.save(output_path)
        print(f"Saved {number}/{NUM_EXAMPLES}: {output_path}")

    print(f"\nCreated {NUM_EXAMPLES} test comparison images.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
