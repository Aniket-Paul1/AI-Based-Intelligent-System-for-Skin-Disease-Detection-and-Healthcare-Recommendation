# ml_code/predict_super_ensemble.py
"""
Predict wrapper that uses canonical config paths.

- Loads Keras CNN (cnn_model.h5) if present.
- If ml_code.extract_embeddings.compute_embedding_for_pil is available, uses embeddings
  and applies RF/XGB predict_proba if those models exist.
- Optionally loads a classes mapping from classes.json or classes.joblib and maps indices robustly.
- Averages available probability vectors (simple ensemble).
"""

from pathlib import Path
import numpy as np
import warnings
import json

# canonical paths
from ml_code.config import CNN_MODEL, RF_MODEL, XGB_MODEL, CLIP_PTH, CLASSES_JSON, IMG_SIZE, MODELS_DIR
import joblib

# lazy model holders
keras_model = None
rf_model = None
xgb_model = None
clip_ckpt = None
CLASSES_RAW = None  # raw loaded object (list or dict or other)
LABELS_BY_INDEX = None  # canonical list mapping index -> label


def load_classes():
    """
    Load class labels from classes.json or classes.joblib (if present).
    Accepts either:
      - list: ["Acne","Vitiligo",...]
      - dict mapping label->index: {"Acne":0, "Vitiligo":1}
      - dict mapping index->label: {"0":"Acne","1":"Vitiligo"}
      - joblib-stored list/dict
    Produces LABELS_BY_INDEX: list such that LABELS_BY_INDEX[i] -> label for index i.
    """
    global CLASSES_RAW, LABELS_BY_INDEX
    LABELS_BY_INDEX = None
    CLASSES_RAW = None

    # Try JSON first
    if CLASSES_JSON.exists():
        try:
            CLASSES_RAW = json.loads(CLASSES_JSON.read_text())
        except Exception as e:
            warnings.warn(f"Failed to load classes.json: {e}")
            CLASSES_RAW = None

    # Try joblib fallback
    if CLASSES_RAW is None:
        joblib_file = MODELS_DIR / "classes.joblib"
        if joblib_file.exists():
            try:
                CLASSES_RAW = joblib.load(str(joblib_file))
            except Exception as e:
                warnings.warn(f"Failed to load classes.joblib: {e}")
                CLASSES_RAW = None

    # If still None, leave LABELS_BY_INDEX None
    if CLASSES_RAW is None:
        LABELS_BY_INDEX = None
        return

    # If CLASSES_RAW is a list, we're done
    if isinstance(CLASSES_RAW, (list, tuple)):
        LABELS_BY_INDEX = list(CLASSES_RAW)
        return

    # If CLASSES_RAW is a dict, try to infer mapping direction
    if isinstance(CLASSES_RAW, dict):
        # Check if dict maps label->index (values likely int)
        all_vals_int = all(isinstance(v, int) or (isinstance(v, str) and v.isdigit()) for v in CLASSES_RAW.values())
        all_keys_int = all(str(k).isdigit() for k in CLASSES_RAW.keys())

        if all_vals_int and not all_keys_int:
            # label -> index : build reverse mapping
            max_index = max(int(v) for v in CLASSES_RAW.values())
            labels = [None] * (max_index + 1)
            for label, idx in CLASSES_RAW.items():
                idx_i = int(idx)
                labels[idx_i] = str(label)
            LABELS_BY_INDEX = labels
            return

        if all_keys_int:
            # keys are indices -> values are labels
            # convert keys to int and place accordingly
            items = [(int(k), v) for k, v in CLASSES_RAW.items()]
            max_index = max(i for i, _ in items)
            labels = [None] * (max_index + 1)
            for i, v in items:
                labels[i] = str(v)
            LABELS_BY_INDEX = labels
            return

    # Else: unknown format -> attempt to coerce to list of strings
    try:
        LABELS_BY_INDEX = [str(x) for x in CLASSES_RAW]
        return
    except Exception:
        LABELS_BY_INDEX = None
        return


def _load_keras():
    global keras_model
    if keras_model is None and CNN_MODEL.exists():
        try:
            from tensorflow.keras.models import load_model
            keras_model = load_model(str(CNN_MODEL))
            print("Loaded Keras model:", CNN_MODEL)
        except Exception as e:
            warnings.warn(f"Failed to load Keras model: {e}")


def _load_rf_xgb():
    global rf_model, xgb_model
    if rf_model is None and RF_MODEL.exists():
        try:
            rf_model = joblib.load(str(RF_MODEL))
            print("Loaded RF model:", RF_MODEL)
        except Exception as e:
            warnings.warn(f"Failed to load RF model: {e}")
    if xgb_model is None and XGB_MODEL.exists():
        try:
            xgb_model = joblib.load(str(XGB_MODEL))
            print("Loaded XGB model:", XGB_MODEL)
        except Exception as e:
            warnings.warn(f"Failed to load XGB model: {e}")


def _load_clip_ckpt():
    global clip_ckpt
    if clip_ckpt is None and CLIP_PTH.exists():
        try:
            import torch
            clip_ckpt = torch.load(str(CLIP_PTH), map_location="cpu")
            print("Loaded CLIP checkpoint:", CLIP_PTH)
        except Exception as e:
            warnings.warn(f"Failed to load CLIP checkpoint: {e}")


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _preprocess_pil(img, size=IMG_SIZE):
    # local preprocessing for keras
    from PIL import Image
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(size)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)


def available_embedding_fn():
    try:
        from ml_code.extract_embeddings import compute_embedding_for_pil
        return compute_embedding_for_pil
    except Exception:
        return None


def predict_from_pil(img):
    """
    Input: PIL.Image
    Returns: dict {label, confidence, confidences}
    """
    # Ensure classes loaded
    load_classes()

    # lazy load models
    _load_keras()
    _load_rf_xgb()
    _load_clip_ckpt()

    preds = []
    n_classes = None

    # Keras path
    if keras_model is not None:
        x = _preprocess_pil(img, size=IMG_SIZE)
        out = keras_model.predict(x).reshape(-1)
        # apply softmax if needed
        if not (out.min() >= 0 and out.max() <= 1):
            out = softmax(out)
        preds.append(out)
        n_classes = out.shape[0]

    # embedding-based models
    emb_fn = available_embedding_fn()
    features = None
    if emb_fn is not None and (rf_model is not None or xgb_model is not None):
        try:
            features = emb_fn(img)  # expected 1D numpy array
            features = np.asarray(features).reshape(-1)
        except Exception as e:
            warnings.warn(f"Embedding computation failed: {e}")
            features = None

    if rf_model is not None and features is not None:
        try:
            proba = rf_model.predict_proba(features.reshape(1, -1)).reshape(-1)
            preds.append(proba)
            n_classes = proba.shape[0] if n_classes is None else n_classes
        except Exception as e:
            warnings.warn(f"RF predict_proba failed: {e}")

    if xgb_model is not None and features is not None:
        try:
            proba = xgb_model.predict_proba(features.reshape(1, -1)).reshape(-1)
            preds.append(proba)
            n_classes = proba.shape[0] if n_classes is None else n_classes
        except Exception as e:
            warnings.warn(f"XGB predict_proba failed: {e}")

    # CLIP: optional; we do not implement a full CLIP inference here

    if not preds:
        raise RuntimeError("No models produced predictions. Ensure your models exist and are loadable.")

    # Normalize and average
    normalized = []
    for p in preds:
        p = np.asarray(p).astype(float)
        if n_classes is not None and p.shape[0] != n_classes:
            if p.shape[0] < n_classes:
                pad = np.full((n_classes - p.shape[0],), 1e-8)
                p = np.concatenate([p, pad])
            else:
                p = p[:n_classes]
        if not np.all((p >= 0) & (p <= 1)):
            p = softmax(p)
        s = p.sum()
        if s <= 0:
            p = np.full(n_classes, 1.0 / n_classes)
        else:
            p = p / s
        normalized.append(p)

    ensemble = np.mean(np.stack(normalized, axis=0), axis=0)

    # Build labels_by_index robustly
    labels = None
    if LABELS_BY_INDEX is not None:
        labels = LABELS_BY_INDEX
    else:
        # if CLASSES_RAW is a mapping label->index, convert it now as a fallback
        if isinstance(CLASSES_RAW, dict):
            try:
                # attempt to build labels list
                items = [(int(v), k) for k, v in CLASSES_RAW.items()]
                max_i = max(i for i, _ in items)
                labels_tmp = [None] * (max_i + 1)
                for i, k in items:
                    labels_tmp[i] = k
                labels = labels_tmp
            except Exception:
                labels = None

    # Final fallback: if still None, create string indexes
    if labels is None:
        labels = [str(i) for i in range(len(ensemble))]

    # Map confidences
    confidences = {}
    for i, score in enumerate(ensemble):
        label = labels[i] if i < len(labels) and labels[i] is not None else str(i)
        confidences[str(label)] = float(score)

    top_idx = int(np.argmax(ensemble))
    top_label = labels[top_idx] if top_idx < len(labels) and labels[top_idx] is not None else str(top_idx)
    top_conf = float(ensemble[top_idx])

    return {"label": top_label, "confidence": round(top_conf, 4), "confidences": {k: round(v, 4) for k, v in confidences.items()}}


if __name__ == "__main__":
    import argparse
    from PIL import Image
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    img = Image.open(args.img)
    result = predict_from_pil(img)

    print("\n=== Prediction Output ===")
    print("Label:", result["label"])
    print("Confidence:", result["confidence"])
    print("\nTop 10 confidences:")
    for k, v in sorted(result["confidences"].items(), key=lambda x: -x[1])[:10]:
        print(k, ":", v)
