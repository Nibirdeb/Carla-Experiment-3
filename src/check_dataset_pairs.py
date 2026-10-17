from pathlib import Path

RGB_DIR = Path("data/rgb")
LANE_DIR = Path("data/lane_masks")
ROAD_DIR = Path("data/road_masks")

rgb_files = {p.name for p in RGB_DIR.glob("*.png")}
lane_files = {p.name for p in LANE_DIR.glob("*.png")}
road_files = {p.name for p in ROAD_DIR.glob("*.png")}

print("=== Dataset Pair Integrity Check ===")
print("RGB images:", len(rgb_files))
print("Lane masks:", len(lane_files))
print("Road masks:", len(road_files))

missing_lane = sorted(rgb_files - lane_files)
missing_road = sorted(rgb_files - road_files)
missing_rgb = sorted((lane_files | road_files) - rgb_files)

print("\nRGB images missing lane masks:", len(missing_lane))
print("Examples:", missing_lane[:10])

print("\nRGB images missing road masks:", len(missing_road))
print("Examples:", missing_road[:10])

print("\nMasks without corresponding RGB images:", len(missing_rgb))
print("Examples:", missing_rgb[:10])

common_files = rgb_files & lane_files & road_files
print("\nComplete image-mask triplets:", len(common_files))

if not missing_lane and not missing_road and not missing_rgb:
    print("\nRESULT: All filenames match.")
else:
    print("\nRESULT: Some filenames do not match. Review the lists above.")