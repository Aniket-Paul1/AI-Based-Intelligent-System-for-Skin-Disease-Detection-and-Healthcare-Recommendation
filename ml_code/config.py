# ml_code/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# =========================
# Project root
# =========================
# ml_code/config.py -> project root is one level up
ROOT = Path(__file__).resolve().parents[1]

# =========================
# Dataset (ALREADY SPLIT)
# =========================
# Your dataset structure:
# ml_code/data/
# ├── train/
# └── val/

DATA_DIR = ROOT / "ml_code" / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

# =========================
# Model artifacts
# =========================
ARTIFACTS_DIR = ROOT / "artifacts"

MODELS_DIR = ARTIFACTS_DIR / "models"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"

# Models
CNN_MODEL = MODELS_DIR / "cnn_model.h5"
CLIP_PTH = MODELS_DIR / "skinclip_finetuned.pth"

# Class mappings
CLASSES_JSON = MODELS_DIR / "classes.json"
CLASSES_JOBLIB = MODELS_DIR / "classes.joblib"

# Embeddings
TRAIN_EMB = EMBEDDINGS_DIR / "train_embeddings.pkl"
VAL_EMB = EMBEDDINGS_DIR / "val_embeddings.pkl"

# =========================
# Training / inference params
# =========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
RANDOM_SEED = 42
