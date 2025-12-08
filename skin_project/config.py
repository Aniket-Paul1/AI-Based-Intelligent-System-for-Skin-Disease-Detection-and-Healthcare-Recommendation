# config.py
from pathlib import Path

# Root of this project (where you run the scripts)
ROOT = Path(__file__).parent.resolve()

# Source images folder where classes are subfolders
SRC_IMAGES = Path(r"C:\Users\ANIKET\OneDrive\Desktop\Adi\Skin disease\skin_project\Datasets\training_images")

# Optional raw images folder (unused by scripts unless you want to copy)
RAW_IMAGES = Path(r"C:\Users\ANIKET\OneDrive\Desktop\Adi\Skin disease\skin_project\Datasets\raw_images\seg_raw_images")

# Where the split train/val will be created (inside project)
DATA_DIR = ROOT / "data"

# Embeddings and models
EMB_DIR = ROOT / "embeddings"
MODELS_DIR = ROOT / "models"

# Image & training params
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
RANDOM_SEED = 42
VAL_SPLIT = 0.20  # fraction for validation
