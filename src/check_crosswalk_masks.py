from pathlib import Path

import cv2
import numpy as np

SAMPLES = ["003022", "004383"]
TASKS = ["lane", "road"]

for sample in SAMPLES:
    print(f"\n=== Sample {sample} ===")

    for task in TASKS:
        path = Path("data") / f"{task}_masks" / f"{sample}.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"{task}: mask could not be read")
            continue

        # Rough lower-image region; includes more than the crosswalk.
        lower_region = mask[300:384, :]

        whole_percentage = np.mean(mask > 0) * 100
        lower_percentage = np.mean(lower_region > 0) * 100

        print(
            f"{task}: "
            f"whole foreground = {whole_percentage:.3f}%, "
            f"lower-region foreground = {lower_percentage:.3f}%"
        )