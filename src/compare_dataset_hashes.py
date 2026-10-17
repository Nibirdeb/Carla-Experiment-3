
"""
Compare dataset files between Experiment 3 and the original CARLA project.

Checks:
- RGB images
- Lane masks
- Road masks
- Matching filenames
- SHA-256 file hashes

This script is read-only and does not modify any files.
"""

from pathlib import Path
import hashlib
import csv


# ============================================================
# Project directories
# ============================================================

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_DIR = Path(
    r"C:\Users\nibir\Desktop\Master_Thesis\Master_Thesis\Carla Experiments"
)

EXPERIMENT_DATA = EXPERIMENT_DIR / "data"
ORIGINAL_DATA = ORIGINAL_DIR / "data"

FOLDERS = [
    "rgb",
    "lane_masks",
    "road_masks",
]

# Save the comparison report in Experiment 3.
REPORT_DIR = EXPERIMENT_DIR / "outputs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_CSV = REPORT_DIR / "dataset_hash_comparison.csv"
REPORT_TXT = REPORT_DIR / "dataset_hash_comparison.txt"


# ============================================================
# SHA-256 helper
# ============================================================

def calculate_sha256(file_path):
    """Calculate a file's SHA-256 hash without modifying it."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# Compare one folder
# ============================================================

def compare_folder(folder_name):
    experiment_folder = EXPERIMENT_DATA / folder_name
    original_folder = ORIGINAL_DATA / folder_name

    print("\n" + "=" * 65)
    print(f"Comparing folder: {folder_name}")
    print("=" * 65)

    if not experiment_folder.is_dir():
        print(f"ERROR: Experiment folder not found: {experiment_folder}")
        return [], {
            "folder": folder_name,
            "experiment_count": 0,
            "original_count": 0,
            "matching_hashes": 0,
            "different_hashes": 0,
            "missing_in_original": 0,
            "missing_in_experiment": 0,
            "error": "Experiment folder missing",
        }

    if not original_folder.is_dir():
        print(f"ERROR: Original folder not found: {original_folder}")
        return [], {
            "folder": folder_name,
            "experiment_count": 0,
            "original_count": 0,
            "matching_hashes": 0,
            "different_hashes": 0,
            "missing_in_original": 0,
            "missing_in_experiment": 0,
            "error": "Original folder missing",
        }

    experiment_files = {
        path.name: path
        for path in experiment_folder.glob("*.png")
        if path.is_file()
    }

    original_files = {
        path.name: path
        for path in original_folder.glob("*.png")
        if path.is_file()
    }

    experiment_names = set(experiment_files)
    original_names = set(original_files)

    common_names = sorted(experiment_names & original_names)
    missing_in_original = sorted(experiment_names - original_names)
    missing_in_experiment = sorted(original_names - experiment_names)

    rows = []
    matching_hashes = 0
    different_hashes = 0

    print(f"Experiment PNG files: {len(experiment_files)}")
    print(f"Original PNG files:   {len(original_files)}")
    print(f"Common filenames:     {len(common_names)}")
    print(f"Missing in original:  {len(missing_in_original)}")
    print(f"Missing in experiment:{len(missing_in_experiment)}")

    for filename in common_names:
        experiment_path = experiment_files[filename]
        original_path = original_files[filename]

        try:
            experiment_hash = calculate_sha256(experiment_path)
            original_hash = calculate_sha256(original_path)

            hashes_match = experiment_hash == original_hash

            if hashes_match:
                status = "MATCH"
                matching_hashes += 1
            else:
                status = "HASH_MISMATCH"
                different_hashes += 1

            rows.append({
                "folder": folder_name,
                "filename": filename,
                "experiment_sha256": experiment_hash,
                "original_sha256": original_hash,
                "status": status,
            })

        except OSError as error:
            rows.append({
                "folder": folder_name,
                "filename": filename,
                "experiment_sha256": "",
                "original_sha256": "",
                "status": f"ERROR: {error}",
            })

        if len(rows) % 500 == 0:
            print(f"Compared {len(rows)} files so far...")

    # Record filenames that do not have a counterpart.
    for filename in missing_in_original:
        rows.append({
            "folder": folder_name,
            "filename": filename,
            "experiment_sha256": "",
            "original_sha256": "",
            "status": "MISSING_IN_ORIGINAL",
        })

    for filename in missing_in_experiment:
        rows.append({
            "folder": folder_name,
            "filename": filename,
            "experiment_sha256": "",
            "original_sha256": "",
            "status": "MISSING_IN_EXPERIMENT",
        })

    summary = {
        "folder": folder_name,
        "experiment_count": len(experiment_files),
        "original_count": len(original_files),
        "matching_hashes": matching_hashes,
        "different_hashes": different_hashes,
        "missing_in_original": len(missing_in_original),
        "missing_in_experiment": len(missing_in_experiment),
        "error": "",
    }

    print(f"Identical files:      {matching_hashes}")
    print(f"Different hashes:     {different_hashes}")

    return rows, summary


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 65)
    print("DATASET SHA-256 COMPARISON")
    print("=" * 65)

    print(f"Experiment 3: {EXPERIMENT_DIR}")
    print(f"Original:     {ORIGINAL_DIR}")

    all_rows = []
    summaries = []

    for folder_name in FOLDERS:
        rows, summary = compare_folder(folder_name)
        all_rows.extend(rows)
        summaries.append(summary)

    # Save detailed CSV report.
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "folder",
            "filename",
            "experiment_sha256",
            "original_sha256",
            "status",
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Save human-readable summary.
    with open(REPORT_TXT, "w", encoding="utf-8") as file:
        file.write("DATASET SHA-256 COMPARISON REPORT\n")
        file.write("=" * 50 + "\n\n")
        file.write(f"Experiment 3: {EXPERIMENT_DIR}\n")
        file.write(f"Original:     {ORIGINAL_DIR}\n\n")

        for summary in summaries:
            file.write(f"Folder: {summary['folder']}\n")
            file.write(
                f"  Experiment PNG files: {summary['experiment_count']}\n"
            )
            file.write(
                f"  Original PNG files:   {summary['original_count']}\n"
            )
            file.write(
                f"  Identical hashes:     {summary['matching_hashes']}\n"
            )
            file.write(
                f"  Different hashes:    {summary['different_hashes']}\n"
            )
            file.write(
                f"  Missing in original:  {summary['missing_in_original']}\n"
            )
            file.write(
                f"  Missing in experiment:{summary['missing_in_experiment']}\n"
            )

            if summary["error"]:
                file.write(f"  ERROR: {summary['error']}\n")

            file.write("\n")

        file.write(
            "Note: matching SHA-256 hashes establish byte-for-byte "
            "identity for the compared files. They do not establish "
            "semantic correctness of the labels.\n"
        )

    # Final summary.
    print("\n" + "=" * 65)
    print("FINAL SUMMARY")
    print("=" * 65)

    for summary in summaries:
        print(
            f"{summary['folder']}: "
            f"{summary['matching_hashes']} identical, "
            f"{summary['different_hashes']} different, "
            f"{summary['missing_in_original']} missing in original, "
            f"{summary['missing_in_experiment']} missing in experiment"
        )

        if summary["error"]:
            print(f"  ERROR: {summary['error']}")

    print("\nReports saved:")
    print(REPORT_CSV)
    print(REPORT_TXT)

    print("\nComparison finished. No dataset files were modified.")


if __name__ == "__main__":
    main()
