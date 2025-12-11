# ml_code/config.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Repo root (one level up from ml_code/)
ROOT = Path(__file__).resolve().parents[1]

# Canonical artifacts dir (default: top-level ./artifacts)
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT / "artifacts")))

# Subfolders inside artifacts
MODELS_DIR = ARTIFACTS_DIR / "models"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "ml_code" / "Datasets")))

# Frequently used artifact paths (convenience)
CNN_MODEL = MODELS_DIR / "cnn_model.h5"
RF_MODEL = MODELS_DIR / "randomforest.joblib"
XGB_MODEL = MODELS_DIR / "xgboost.joblib"
CLIP_PTH = MODELS_DIR / "skinclip_finetuned.pth"
CLASSES_JSON = MODELS_DIR / "classes.json"
TRAIN_EMB = EMBEDDINGS_DIR / "train_embeddings.pkl"
VAL_EMB = EMBEDDINGS_DIR / "val_embeddings.pkl"

# Image and training defaults
# IMG_SIZE can be "224,224" in env or default uses 224x224
IMG_SIZE = tuple(int(x) for x in os.getenv("IMG_SIZE", "224,224").split(","))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))