# batch_test_raw.py
import csv
from pathlib import Path
import json
import numpy as np
import joblib
from tqdm import tqdm
from config import RAW_IMAGES, MODELS_DIR, IMG_SIZE
import warnings
warnings.filterwarnings("ignore")

# tensorflow imports for cnn & embedding extractor
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Helper: build embedding extractor (MobileNetV2->globalpool)
def build_emb_model():
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    gap = GlobalAveragePooling2D()(base.output)
    model = Model(inputs=base.input, outputs=gap)
    return model

# Load available models (if present)
def load_models():
    models = {}
    # CNN
    cnn_path = MODELS_DIR / "cnn_model.h5"
    classes_json = MODELS_DIR / "classes.json"
    if cnn_path.exists() and classes_json.exists():
        try:
            cnn = tf.keras.models.load_model(str(cnn_path))
            with open(classes_json, "r") as f:
                cnn_classes = json.load(f)
            models["cnn"] = {"model": cnn, "classes": cnn_classes}
            print("Loaded CNN model.")
        except Exception as e:
            print("Error loading CNN:", e)

    # RF / XGB
    rf_path = MODELS_DIR / "randomforest.joblib"
    xgb_path = MODELS_DIR / "xgboost.joblib"
    classes_job = MODELS_DIR / "classes.joblib"
    if classes_job.exists():
        classes_list = joblib.load(classes_job)
    else:
        classes_list = None

    if rf_path.exists():
        try:
            rf = joblib.load(rf_path)
            models["rf"] = {"model": rf, "classes": classes_list}
            print("Loaded RandomForest.")
        except Exception as e:
            print("Error loading RandomForest:", e)

    if xgb_path.exists():
        try:
            xgb_clf = joblib.load(xgb_path)
            models["xgb"] = {"model": xgb_clf, "classes": classes_list}
            print("Loaded XGBoost.")
        except Exception as e:
            print("Error loading XGBoost:", e)

    return models

# Preprocess image for CNN
def prep_for_cnn(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

# Preprocess image for embedding-based models
def prep_for_emb(img_path):
    img = load_img(img_path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

# Top-k helper
def top_k_labels(probs, classes_map, k=3):
    idxs = np.argsort(probs)[-k:][::-1]
    results = []
    for i in idxs:
        label = None
        if isinstance(classes_map, dict):
            # classes_map: {index_str: label}
            label = classes_map.get(str(i), classes_map.get(i, str(i)))
        elif isinstance(classes_map, (list, tuple, np.ndarray)):
            label = classes_map[i]
        else:
            label = str(i)
        results.append(f"{label}:{float(probs[i]):.4f}")
    return ";".join(results)

def find_images(root: Path):
    exts = (".jpg", ".jpeg", ".png")
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def main():
    RAW = RAW_IMAGES
    if not RAW.exists():
        raise SystemExit(f"RAW_IMAGES path not found: {RAW}")
    image_paths = find_images(RAW)
    if not image_paths:
        raise SystemExit(f"No images found under RAW_IMAGES: {RAW}")
    print(f"Found {len(image_paths)} images in RAW_IMAGES.")

    models = load_models()
    if not models:
        raise SystemExit("No models found in MODELS_DIR. Run training scripts first.")

    # build emb model only if rf/xgb present
    emb_model = None
    if "rf" in models or "xgb" in models:
        emb_model = build_emb_model()

    out_csv = Path("batch_predictions_raw.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        header = [
            "image_path",
            "model_type",
            "pred_label",
            "confidence",
            "top3"
        ]
        writer.writerow(header)

        for p in tqdm(image_paths):
            pstr = str(p)
            # CNN
            if "cnn" in models:
                try:
                    arr = prep_for_cnn(pstr)
                    preds = models["cnn"]["model"].predict(arr, verbose=0)[0]
                    idx = int(np.argmax(preds))
                    classes_map = models["cnn"]["classes"]
                    label = classes_map.get(str(idx), list(classes_map.values())[idx]) if classes_map else str(idx)
                    top3 = top_k_labels(preds, classes_map)
                    writer.writerow([pstr, "cnn", label, f"{float(preds[idx]):.6f}", top3])
                except Exception as e:
                    writer.writerow([pstr, "cnn", "ERROR", str(e), ""])

            # RF
            if "rf" in models:
                try:
                    arr = prep_for_emb(pstr)
                    feat = emb_model.predict(arr, verbose=0)
                    probs = models["rf"]["model"].predict_proba(feat)[0]
                    idx = int(np.argmax(probs))
                    classes_map = models["rf"]["classes"]
                    label = classes_map[idx] if isinstance(classes_map, (list,tuple)) else str(idx)
                    top3 = top_k_labels(probs, classes_map)
                    writer.writerow([pstr, "rf", label, f"{float(probs[idx]):.6f}", top3])
                except Exception as e:
                    writer.writerow([pstr, "rf", "ERROR", str(e), ""])

            # XGB
            if "xgb" in models:
                try:
                    arr = prep_for_emb(pstr)
                    feat = emb_model.predict(arr, verbose=0)
                    probs = models["xgb"]["model"].predict_proba(feat)[0]
                    idx = int(np.argmax(probs))
                    classes_map = models["xgb"]["classes"]
                    label = classes_map[idx] if isinstance(classes_map, (list,tuple)) else str(idx)
                    top3 = top_k_labels(probs, classes_map)
                    writer.writerow([pstr, "xgb", label, f"{float(probs[idx]):.6f}", top3])
                except Exception as e:
                    writer.writerow([pstr, "xgb", "ERROR", str(e), ""])

    print("Batch predictions finished. CSV:", out_csv.resolve())

if __name__ == "__main__":
    main()
