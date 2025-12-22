# ml_code/extract_embeddings.py

import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from ml_code.config import TRAIN_DIR, VAL_DIR, EMBEDDINGS_DIR, IMG_SIZE

# -------------------------------
# Model: MobileNetV2 as feature extractor
# -------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)

# -------------------------------
# Utility: load and preprocess image
# -------------------------------
def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


# -------------------------------
# Extract embeddings from a directory
# -------------------------------
def extract_from_dir(root_dir):
    features = []
    labels = []

    class_names = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])

    for idx, class_name in enumerate(class_names):
        class_dir = root_dir / class_name
        image_files = list(class_dir.glob("*"))

        for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
            try:
                img_arr = load_image(img_path)
                emb = base_model.predict(img_arr, verbose=0)[0]
                features.append(emb)
                labels.append(idx)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    return np.array(features), np.array(labels), class_names


# -------------------------------
# Main execution
# -------------------------------
def main():
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    print("Extracting TRAIN embeddings...")
    X_train, y_train, class_names = extract_from_dir(TRAIN_DIR)

    print("Extracting VAL embeddings...")
    X_val, y_val, _ = extract_from_dir(VAL_DIR)

    # Save embeddings
    joblib.dump((X_train, y_train), EMBEDDINGS_DIR / "train_embeddings.pkl")
    joblib.dump((X_val, y_val), EMBEDDINGS_DIR / "val_embeddings.pkl")

    print("Embeddings saved to:", EMBEDDINGS_DIR)
    print("Classes:", class_names)


if __name__ == "__main__":
    main()
