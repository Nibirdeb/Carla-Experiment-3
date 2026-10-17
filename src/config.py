from pathlib import Path
import torch


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

RGB_DIR = DATA_DIR / "rgb"
LANE_MASK_DIR = DATA_DIR / "lane_masks"
ROAD_MASK_DIR = DATA_DIR / "road_masks"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================================
# Dataset
# ============================================================

IMAGE_WIDTH = 1248
IMAGE_HEIGHT = 384

NUM_CLASSES = 2


# ============================================================
# Training
# ============================================================

BATCH_SIZE = 4
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4

NUM_WORKERS = 0


# ============================================================
# Device
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")