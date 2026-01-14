from pathlib import Path
import json

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = DATA_PROCESSED / "train"
VAL_DIR   = DATA_PROCESSED / "val"
TEST_DIR  = DATA_PROCESSED / "test"   # currently empty, but kept for future

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CLASS_INDEX_PATH = ARTIFACTS_DIR / "class_index.json"
MODEL_WEIGHTS_PATH = ARTIFACTS_DIR / "model_efficientnet_b0.pth"

# Load class info (15 classes after filtering to 4 crops)
with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
    CLASS_INFO = json.load(f)

NUM_CLASSES = CLASS_INFO["num_classes"]

# Training hyperparameters (tuned for GTX 1650 4GB)
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2          # can try 4 if system is free
EPOCHS = 20              # max epochs
EARLY_STOP_PATIENCE = 4  # early stopping patience
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

MODEL_NAME = "efficientnet_b0"  # torchvision model
DEVICE = "cuda"  # train.py will fall back to CPU if GPU not available
