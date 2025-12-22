import json
import numpy as np
from pathlib import Path
from PIL import Image

import joblib
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from ml_code.config import (
    CNN_MODEL,
    RF_MODEL,
    XGB_MODEL,
    CLASSES_JSON,
    IMG_SIZE,
)

# ======================================================
# Global cached models
# ======================================================
keras_model = None
rf_model = None
xgb_model = None
embedding_model = None
LABELS_BY_INDEX = None


# ======================================================
# Load class labels
# ======================================================
def load_classes():
    global LABELS_BY_INDEX

    if LABELS_BY_INDEX is not None:
        return

    if not CLASSES_JSON.exists():
        raise FileNotFoundError("classes.json not found")

    with open(CLASSES_JSON, "r") as f:
        class_to_idx = json.load(f)

    LABELS_BY_INDEX = [None] * len(class_to_idx)
    for label, idx in class_to_idx.items():
        LABELS_BY_INDEX[int(idx)] = label


# ======================================================
# Preprocess image for CNN
# ======================================================
def preprocess_for_cnn(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


# ======================================================
# Load models (lazy loading)
# ======================================================
def load_models():
    global keras_model, rf_model, xgb_model, embedding_model

    if keras_model is None:
        keras_model = tf.keras.models.load_model(str(CNN_MODEL))
        print("Loaded Keras model:", CNN_MODEL)

    if rf_model is None and RF_MODEL.exists():
        rf_model = joblib.load(RF_MODEL)
        print("Loaded RF model:", RF_MODEL)

    if xgb_model is None and XGB_MODEL.exists():
        xgb_model = joblib.load(XGB_MODEL)
        print("Loaded XGB model:", XGB_MODEL)

    if embedding_model is None:
        embedding_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",              # 1280-D embedding
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        )


# ======================================================
# Compute 1280-D embedding
# ======================================================
def compute_embedding(img):
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    emb = embedding_model.predict(arr, verbose=0)
    return emb  # shape (1, 1280)


# ======================================================
# Weighted ensemble
# ======================================================
def weighted_ensemble(preds, weights):
    ensemble = None
    for key, prob in preds.items():
        if prob is None:
            continue
        w = weights.get(key, 0.0)
        if ensemble is None:
            ensemble = w * prob
        else:
            ensemble += w * prob
    return ensemble


# ======================================================
# Main prediction logic (CORE)
# ======================================================
def predict_from_pil(img):
    """
    Returns:
    {
        "label": str,
        "confidence": float,
        "confidences": dict[label -> prob]
    }
    """

    load_classes()
    load_models()

    preds = {}

    # -----------------------------
    # CNN prediction
    # -----------------------------
    x = preprocess_for_cnn(img)
    cnn_probs = keras_model.predict(x, verbose=0)[0]
    cnn_probs = cnn_probs / cnn_probs.sum()
    preds["cnn"] = cnn_probs

    # -----------------------------
    # RF / XGB prediction
    # -----------------------------
    features = compute_embedding(img)

    if rf_model is not None:
        rf_probs = rf_model.predict_proba(features)[0]
        rf_probs = rf_probs / rf_probs.sum()
        preds["rf"] = rf_probs

    if xgb_model is not None:
        xgb_probs = xgb_model.predict_proba(features)[0]
        xgb_probs = xgb_probs / xgb_probs.sum()
        preds["xgb"] = xgb_probs

    # -----------------------------
    # Ensemble weights
    # -----------------------------
    weights = {
        "cnn": 0.70,
        "rf": 0.15,
        "xgb": 0.15,
    }

    # -----------------------------
    # Weighted ensemble
    # -----------------------------
    ensemble = weighted_ensemble(preds, weights)
    ensemble = ensemble / ensemble.sum()

    # -----------------------------
    # Output formatting
    # -----------------------------
    confidences = {
        LABELS_BY_INDEX[i]: float(ensemble[i])
        for i in range(len(ensemble))
    }

    top_idx = int(np.argmax(ensemble))

    return {
        "label": LABELS_BY_INDEX[top_idx],
        "confidence": round(float(ensemble[top_idx]), 4),
        "confidences": {
            k: round(v, 4) for k, v in confidences.items()
        },
    }


# ======================================================
# Streamlit-compatible wrapper (IMPORTANT)
# ======================================================
def predict_image(img):
    """
    Wrapper for Streamlit UI.

    Returns:
        label (str)
        confidence (float)        # 0–1
        top_probabilities (dict)
    """

    result = predict_from_pil(img)

    label = result["label"]
    confidence = result["confidence"]
    top_probabilities = result["confidences"]

    return label, confidence, top_probabilities


# ======================================================
# CLI runner (UNCHANGED)
# ======================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to image")
    args = parser.parse_args()

    image = Image.open(args.img)
    result = predict_from_pil(image)

    print("\n=== Prediction Output ===")
    print("Label:", result["label"])
    print("Confidence:", result["confidence"])
    print("\nTop probabilities:")
    for k, v in sorted(
        result["confidences"].items(),
        key=lambda x: -x[1]
    )[:10]:
        print(f"{k}: {v}")
