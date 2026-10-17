
from pathlib import Path

import cv2
import numpy as np

SEMANTIC_DIR = Path(
    r"C:\Users\nibir\Desktop\Master_Thesis\Master_Thesis\Carla Experiments\data\semantic"
)

files = sorted(SEMANTIC_DIR.glob("*.png"))

print("=" * 60)
print("SEMANTIC SAMPLE INSPECTION")
print("=" * 60)
print("Files found:", len(files))

if not files:
    print("No semantic PNG files found.")
    raise SystemExit(1)

for path in files:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    print(f"\nFile: {path.name}")

    if image is None:
        print("ERROR: Could not read image.")
        continue

    print("Shape:", image.shape)
    print("Data type:", image.dtype)

    unique_values = np.unique(image)

    print("Unique pixel values:", len(unique_values))
    print("First 100 unique values:", unique_values[:100])

    print("Contains class ID 1:", bool(np.any(image == 1)))
    print("Contains class ID 24:", bool(np.any(image == 24)))

print("\nInspection complete. No files were modified.")
