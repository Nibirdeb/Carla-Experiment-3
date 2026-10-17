from pathlib import Path

import cv2
import numpy as np

FOLDERS = {
    "lane": Path("data/lane_masks"),
    "road": Path("data/road_masks"),
}

EXPECTED_SHAPE = (384, 1248)

for name, folder in FOLDERS.items():
    files = sorted(folder.glob("*.png"))
    ratios = []
    empty = []
    full = []
    bad = []

    for path in files:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            bad.append(path.name)
            continue

        if mask.shape != EXPECTED_SHAPE:
            bad.append(f"{path.name}: shape={mask.shape}")

        ratio = float(np.count_nonzero(mask > 0) / mask.size)
        ratios.append((ratio, path.name))

        if ratio == 0:
            empty.append(path.name)
        if ratio == 1:
            full.append(path.name)

    print(f"\n=== {name.upper()} MASKS ===")
    print("PNG files:", len(files))
    print("Readable masks:", len(ratios))
    print("Empty masks:", len(empty), empty[:10])
    print("Fully white masks:", len(full), full[:10])
    print("Unreadable or wrong-size masks:", len(bad), bad[:10])

    if ratios:
        values = np.array([ratio for ratio, _ in ratios])

        print(
            "Foreground percentage min/median/max:",
            round(values.min() * 100, 3),
            round(np.median(values) * 100, 3),
            round(values.max() * 100, 3),
        )

        print(
            "Smallest 5:",
            [(filename, round(ratio * 100, 3))
             for ratio, filename in sorted(ratios)[:5]],
        )

        print(
            "Largest 5:",
            [(filename, round(ratio * 100, 3))
             for ratio, filename in sorted(ratios, reverse=True)[:5]],
        )