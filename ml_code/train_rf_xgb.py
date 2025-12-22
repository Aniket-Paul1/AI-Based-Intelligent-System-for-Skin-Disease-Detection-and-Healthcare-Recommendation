# ml_code/train_rf_xgb.py

import joblib
import json
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ml_code.config import (
    TRAIN_EMB,
    VAL_EMB,
    MODELS_DIR,
    CLASSES_JSON
)

# -------------------------------
# Load embeddings
# -------------------------------
def load_embeddings(split: str):
    if split == "train":
        path = TRAIN_EMB
    elif split == "val":
        path = VAL_EMB
    else:
        raise ValueError("split must be 'train' or 'val'")

    data = joblib.load(path)

    # Current format: (X, y)
    if isinstance(data, tuple) and len(data) == 2:
        X, y = data
        return X, y

    # Backward compatibility (old formats)
    if isinstance(data, tuple) and len(data) >= 4:
        X, y = data[0], data[1]
        return X, y

    raise RuntimeError("Unsupported embedding format")


# -------------------------------
# Main training logic
# -------------------------------
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    X_train, y_train = load_embeddings("train")
    X_val, y_val = load_embeddings("val")

    print("Train embeddings:", X_train.shape)
    print("Val embeddings:", X_val.shape)

    # Load class names
    with open(CLASSES_JSON, "r") as f:
        class_to_idx = json.load(f)

    # Build index → class mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # -------------------------------
    # Random Forest
    # -------------------------------
    print("Training RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # -------------------------------
    # XGBoost
    # -------------------------------
    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(idx_to_class),
        random_state=42
    )
    xgb.fit(X_train, y_train)

    # -------------------------------
    # Save models
    # -------------------------------
    joblib.dump(rf, MODELS_DIR / "randomforest.joblib")
    joblib.dump(xgb, MODELS_DIR / "xgboost.joblib")
    joblib.dump(idx_to_class, MODELS_DIR / "classes.joblib")

    print("RF and XGBoost models saved.")
    print("Classes:", idx_to_class)


if __name__ == "__main__":
    main()
