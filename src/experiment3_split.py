
from pathlib import Path
import csv

from config import RGB_DIR, LANE_MASK_DIR, ROAD_MASK_DIR

# Ordered, contiguous split of the 5,000 numbered samples.
SPLITS = {
    "train": (1, 3500),
    "val": (3501, 4250),
    "test": (4251, 5000),
}


def main():
    project_dir = Path(__file__).resolve().parent.parent
    output_dir = project_dir / "experiment3_controlled"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    errors = []

    for split_name, (start_id, end_id) in SPLITS.items():
        for sample_id in range(start_id, end_id + 1):
            filename = f"{sample_id:06d}.png"

            rgb = Path(RGB_DIR) / filename
            lane = Path(LANE_MASK_DIR) / filename
            road = Path(ROAD_MASK_DIR) / filename

            missing = [
                str(path)
                for path in (rgb, lane, road)
                if not path.is_file()
            ]

            if missing:
                errors.append(
                    f"{split_name} {filename}: missing {missing}"
                )
                continue

            rows.append({
                "split": split_name,
                "sample_id": sample_id,
                "filename": filename,
                "rgb_path": str(rgb),
                "lane_mask_path": str(lane),
                "road_mask_path": str(road),
            })

    if errors:
        print("ERROR: Dataset files are missing:")
        for error in errors[:30]:
            print(error)
        raise SystemExit(
            f"Found {len(errors)} samples with missing files. "
            "No split manifest was written."
        )

    manifest = output_dir / "split_manifest.csv"

    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split",
                "sample_id",
                "filename",
                "rgb_path",
                "lane_mask_path",
                "road_mask_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest: {manifest}")
    for split_name, (start_id, end_id) in SPLITS.items():
        count = sum(row["split"] == split_name for row in rows)
        print(
            f"{split_name}: IDs {start_id:06d}-{end_id:06d}, "
            f"{count} samples"
        )

    print(f"Total verified samples: {len(rows)}")
    print("This script only checks files and writes a CSV manifest.")
    print("It does not train a model or modify the original dataset.")


if __name__ == "__main__":
    main()

