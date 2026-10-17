from pathlib import Path
from statistics import median
from PIL import Image
import numpy as np

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent

RGB_DIR = PROJECT_DIR / "data" / "rgb"
LANE_DIR = PROJECT_DIR / "data" / "lane_masks"
ROAD_DIR = PROJECT_DIR / "data" / "road_masks"
OUTPUT_DIR = PROJECT_DIR / "outputs"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def get_image_files(folder):
    """Return image files in a folder, sorted by filename."""
    if not folder.exists():
        print(f"WARNING: Folder does not exist: {folder}")
        return []

    return sorted(
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def inspect_masks(files, mask_name):
    """Check readability, dimensions, and foreground pixel statistics."""
    foreground_percentages = []
    empty_count = 0
    full_white_count = 0
    unreadable_count = 0
    wrong_size_count = 0
    readable_count = 0
    dimensions = set()

    for file in files:
        try:
            with Image.open(file) as image:
                image.load()
                mask = np.array(image.convert("L"))

            readable_count += 1
            dimensions.add(f"{mask.shape[1]}x{mask.shape[0]}")

            if mask.shape != (384, 1248):
                wrong_size_count += 1

            foreground = mask > 0
            foreground_count = int(np.count_nonzero(foreground))
            total_pixels = mask.size

            percentage = foreground_count / total_pixels * 100
            foreground_percentages.append(percentage)

            if foreground_count == 0:
                empty_count += 1

            if foreground_count == total_pixels:
                full_white_count += 1

        except Exception as error:
            unreadable_count += 1
            print(f"WARNING: Cannot read {mask_name} {file.name}: {error}")

    print(f"\n{mask_name.upper()} MASK REPORT")
    print("-" * 45)
    print(f"Files found:              {len(files)}")
    print(f"Readable masks:           {readable_count}")
    print(f"Unreadable masks:         {unreadable_count}")
    print(f"Unexpected dimensions:    {wrong_size_count}")
    print(f"Empty masks:              {empty_count}")
    print(f"Fully white masks:        {full_white_count}")
    print(f"Dimensions found:         {sorted(dimensions)}")

    if foreground_percentages:
        print(
            f"Foreground percentage min:    "
            f"{min(foreground_percentages):.3f}%"
        )
        print(
            f"Foreground percentage median: "
            f"{median(foreground_percentages):.3f}%"
        )
        print(
            f"Foreground percentage max:    "
            f"{max(foreground_percentages):.3f}%"
        )

    return {
        "files": len(files),
        "readable": readable_count,
        "unreadable": unreadable_count,
        "wrong_size": wrong_size_count,
        "empty": empty_count,
        "full_white": full_white_count,
        "dimensions": sorted(dimensions),
        "foreground_min": (
            min(foreground_percentages)
            if foreground_percentages else None
        ),
        "foreground_median": (
            median(foreground_percentages)
            if foreground_percentages else None
        ),
        "foreground_max": (
            max(foreground_percentages)
            if foreground_percentages else None
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rgb_files = get_image_files(RGB_DIR)
    lane_files = get_image_files(LANE_DIR)
    road_files = get_image_files(ROAD_DIR)

    rgb_names = {file.stem for file in rgb_files}
    lane_names = {file.stem for file in lane_files}
    road_names = {file.stem for file in road_files}

    missing_lane = sorted(rgb_names - lane_names)
    missing_road = sorted(rgb_names - road_names)
    lane_without_rgb = sorted(lane_names - rgb_names)
    road_without_rgb = sorted(road_names - rgb_names)

    report_lines = []

    def record(line=""):
        print(line)
        report_lines.append(line)

    record("DATASET QUALITY REPORT")
    record("=" * 60)
    record(f"Project directory: {PROJECT_DIR}")
    record(f"RGB directory:     {RGB_DIR}")
    record(f"Lane masks:        {LANE_DIR}")
    record(f"Road masks:        {ROAD_DIR}")
    record()

    record("FILE COUNTS")
    record("-" * 45)
    record(f"RGB images:        {len(rgb_files)}")
    record(f"Lane masks:        {len(lane_files)}")
    record(f"Road masks:        {len(road_files)}")
    record(f"Complete triplets: {len(rgb_names & lane_names & road_names)}")
    record()

    record("FILENAME MATCHING")
    record("-" * 45)
    record(f"RGB images missing lane masks: {len(missing_lane)}")
    record(f"RGB images missing road masks: {len(missing_road)}")
    record(f"Lane masks without RGB:        {len(lane_without_rgb)}")
    record(f"Road masks without RGB:        {len(road_without_rgb)}")

    if missing_lane:
        record(f"Missing lane examples: {missing_lane[:20]}")

    if missing_road:
        record(f"Missing road examples: {missing_road[:20]}")

    if lane_without_rgb:
        record(f"Lane masks without RGB examples: {lane_without_rgb[:20]}")

    if road_without_rgb:
        record(f"Road masks without RGB examples: {road_without_rgb[:20]}")

    record()

    lane_stats = inspect_masks(lane_files, "lane")
    road_stats = inspect_masks(road_files, "road")

    # Add mask statistics to the saved report.
    for name, stats in [("LANE", lane_stats), ("ROAD", road_stats)]:
        report_lines.append("")
        report_lines.append(f"{name} MASK REPORT")
        report_lines.append("-" * 45)

        for key, value in stats.items():
            report_lines.append(f"{key}: {value}")

    record()
    record("INTERPRETATION NOTES")
    record("-" * 45)
    record("- This report checks file integrity and basic mask statistics.")
    record("- It does not prove that mask labels are semantically correct.")
    record("- The original mask-generation procedure has not been verified.")

    report_path = OUTPUT_DIR / "dataset_quality_report.txt"
    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8"
    )

    print("\n" + "=" * 60)
    print(f"Report saved to:\n{report_path}")


if __name__ == "__main__":
    main()