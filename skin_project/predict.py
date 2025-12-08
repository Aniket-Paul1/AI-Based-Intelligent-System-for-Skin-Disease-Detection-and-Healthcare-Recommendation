# predict_ensemble.py
"""
Ensemble predictor with safe probability smoothing (tempering) to reduce overconfident models.
Usage:
    python predict_ensemble.py "C:\...s1.jpg"

Behavior:
 - Loads available models from models/
 - Builds per-model probability vectors aligned to canonical class_list
 - Applies per-model smoothing (alpha power) to probabilities, then renormalizes
 - Averages smoothed probability vectors and outputs ensemble decision
"""

import argparse
from pathlib import Path
import json
import numpy as np
import joblib
import sys
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODELS_DIR = Path("models")
IMG_SIZE = (224, 224)

# Per-model smoothing hyperparams (tuneable)
# alpha < 1 -> flatten / reduce extreme confidences
DEFAULT_ALPHAS = {
    "cnn": 1.0,
    "randomforest": 0.95,
    "xgboost": 0.8,          # aggressive flattening for XGBoost (was overconfident)
    "xgboost_calib": 0.9,    # if you later have a calibrated xgb, use milder smoothing
}

def load_cnn():
    p = MODELS_DIR / "cnn_model.h5"
    cj = MODELS_DIR / "classes.json"
    if not (p.exists() and cj.exists()):
        return None
    model = tf.keras.models.load_model(str(p))
    classes = json.load(open(cj))
    class_list = [classes[str(i)] for i in range(len(classes))]
    return {"model": model, "classes_map": classes, "class_list": class_list}

def load_sklearn_models():
    results = {}
    cls_file = MODELS_DIR / "classes.joblib"
    class_list = joblib.load(cls_file) if cls_file.exists() else None

    # prefer calibrated xgboost if present
    for fname in ["xgboost_calib.joblib", "xgboost.joblib", "randomforest.joblib"]:
        p = MODELS_DIR / fname
        if p.exists():
            key = fname.replace(".joblib", "")
            try:
                results[key] = {"model": joblib.load(p), "class_list": class_list}
            except Exception as e:
                print(f"Warning: failed to load {p}: {e}")
    return results

def build_emb_model():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    gap = GlobalAveragePooling2D()(base.output)
    return Model(inputs=base.input, outputs=gap)

def prep_for_cnn(path):
    arr = img_to_array(load_img(path, target_size=IMG_SIZE))/255.0
    return np.expand_dims(arr, 0)

def prep_for_emb(path):
    arr = img_to_array(load_img(path, target_size=IMG_SIZE))
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)

def normalize_probs(p):
    p = np.asarray(p, dtype=float)
    s = p.sum()
    if s <= 0:
        return np.ones_like(p) / len(p)
    return p / s

def apply_smoothing(probs, alpha):
    # probs -> probs^alpha, renormalize
    probs = np.asarray(probs, dtype=float)
    # avoid negative/NaN
    probs = np.clip(probs, 1e-12, 1.0)
    p2 = np.power(probs, alpha)
    return normalize_probs(p2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="path to image")
    parser.add_argument("--alphas", help="optional JSON string to override per-model alphas", default=None)
    args = parser.parse_args()

    img = Path(args.img)
    if not img.exists():
        print("Image not found:", img); sys.exit(1)

    # Optionally override alphas via CLI
    alphas = DEFAULT_ALPHAS.copy()
    if args.alphas:
        try:
            user_alphas = json.loads(args.alphas)
            alphas.update(user_alphas)
        except Exception:
            print("Couldn't parse --alphas JSON; ignoring.")

    cnn_info = load_cnn()
    skl_models = load_sklearn_models()

    if not cnn_info and not skl_models:
        print("No models available in models/ (cnn_model.h5 or randomforest/xgboost)."); sys.exit(1)

    # canonical class list
    if (MODELS_DIR / "classes.joblib").exists():
        class_list = joblib.load(MODELS_DIR / "classes.joblib")
    elif cnn_info:
        class_list = cnn_info["class_list"]
    else:
        raise SystemExit("No class mapping found. Ensure models/classes.joblib or models/classes.json exists.")

    n_classes = len(class_list)
    emb_model = None
    if skl_models:
        try:
            emb_model = build_emb_model()
        except Exception as e:
            print("Failed to build embedding extractor:", e)
            emb_model = None

    model_probs = {}  # model_name -> aligned prob vector

    # CNN
    if cnn_info:
        arr = prep_for_cnn(str(img))
        preds = cnn_info["model"].predict(arr, verbose=0)[0]
        # align cnn preds into canonical ordering
        aligned = np.zeros(n_classes, dtype=float)
        for idx_str, label in cnn_info["classes_map"].items():
            try:
                idx = int(idx_str)
                pos = class_list.index(label)
                aligned[pos] = float(preds[idx])
            except Exception:
                pass
        aligned = normalize_probs(aligned)
        alpha = alphas.get("cnn", 1.0)
        smoothed = apply_smoothing(aligned, alpha)
        model_probs["cnn"] = smoothed
        print("=== CNN top-3 ===")
        for i in np.argsort(smoothed)[-3:][::-1]:
            print(f"  - {class_list[i]} ({smoothed[i]*100:.2f}%)")

    # sklearn models (xgboost_calib, xgboost, randomforest)
    for name, info in skl_models.items():
        model = info["model"]
        if emb_model is None:
            print(f"Skipping {name}: embedding extractor unavailable.")
            continue
        feat = emb_model.predict(prep_for_emb(str(img)), verbose=0)

        # get probs (predict_proba if available)
        try:
            probs = model.predict_proba(feat)[0]
        except Exception:
            pred = model.predict(feat)
            probs = np.zeros(n_classes, dtype=float)
            probs[int(pred[0])] = 1.0

        probs = normalize_probs(probs)
        # decide alpha: prefer specific calibrated version
        base_key = name.replace(".joblib","")
        if "xgboost_calib" in name:
            alpha = alphas.get("xgboost_calib", 0.9)
        elif "xgboost" in name:
            alpha = alphas.get("xgboost", 0.6)
        else:
            alpha = alphas.get(name, 0.9)

        smoothed = apply_smoothing(probs, alpha)
        model_probs[name] = smoothed
        print(f"=== {name.upper()} top-3 ===")
        for i in np.argsort(smoothed)[-3:][::-1]:
            print(f"  - {class_list[i]} ({smoothed[i]*100:.2f}%)")

    if not model_probs:
        print("No probability vectors available.")
        sys.exit(1)

    # Average smoothed probabilities across models
    all_probs = np.stack(list(model_probs.values()), axis=0)
    avg = np.mean(all_probs, axis=0)
    avg = normalize_probs(avg)
    top_idx = int(np.argmax(avg))
    top_label = class_list[top_idx]
    top_conf = float(avg[top_idx])

    print("\n=====================================")
    print(f"Ensemble detected disease: {top_label} ({top_conf*100:.2f}% confidence)")
    print("Ensemble top-3:")
    for i in np.argsort(avg)[-3:][::-1]:
        print(f"  - {class_list[i]} ({avg[i]*100:.2f}%)")
    print("=====================================\n")

if __name__ == "__main__":
    main()
