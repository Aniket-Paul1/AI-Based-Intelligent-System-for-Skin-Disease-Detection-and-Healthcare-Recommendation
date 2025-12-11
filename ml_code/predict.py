# ml_code/predict.py
"""
Simple prediction wrapper for single-image inference.
Uses canonical config paths from ml_code.config.
"""
from ml_code.config import CNN_MODEL, RF_MODEL, XGB_MODEL, CLASSES_JSON, IMG_SIZE
import json
import numpy as np
import warnings

# lazy loaders
keras_model = None
rf_model = None
xgb_model = None
CLASSES = None

def _load_classes():
    global CLASSES
    if CLASSES is None and CLASSES_JSON.exists():
        with open(CLASSES_JSON, "r") as f:
            CLASSES = json.load(f)

def _load_keras():
    global keras_model
    if keras_model is None and CNN_MODEL.exists():
        try:
            from tensorflow.keras.models import load_model
            keras_model = load_model(str(CNN_MODEL))
        except Exception as e:
            warnings.warn(f"Keras load failed: {e}")

def _load_rf_xgb():
    global rf_model, xgb_model
    try:
        import joblib
        if rf_model is None and RF_MODEL.exists():
            rf_model = joblib.load(str(RF_MODEL))
        if xgb_model is None and XGB_MODEL.exists():
            xgb_model = joblib.load(str(XGB_MODEL))
    except Exception:
        pass

def _preprocess(img_pil, size=IMG_SIZE):
    from PIL import Image
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    img_pil = img_pil.resize(size)
    arr = np.array(img_pil).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

def predict_from_pil(img_pil):
    _load_classes()
    _load_keras()
    _load_rf_xgb()

    preds = []
    n_classes = None

    if keras_model is not None:
        x = _preprocess(img_pil)
        out = keras_model.predict(x).reshape(-1)
        if not (out.min() >= 0 and out.max() <= 1):
            exps = np.exp(out - np.max(out)); out = exps / exps.sum()
        preds.append(out)
        n_classes = out.shape[0]

    # embedding-based models (optional)
    try:
        from ml_code.extract_embeddings import compute_embedding_for_pil
        features = compute_embedding_for_pil(img_pil)
    except Exception:
        features = None

    if features is not None and rf_model is not None:
        try:
            proba = rf_model.predict_proba(features.reshape(1, -1)).reshape(-1)
            preds.append(proba)
            n_classes = proba.shape[0] if n_classes is None else n_classes
        except Exception:
            pass

    if features is not None and xgb_model is not None:
        try:
            proba = xgb_model.predict_proba(features.reshape(1, -1)).reshape(-1)
            preds.append(proba)
            n_classes = proba.shape[0] if n_classes is None else n_classes
        except Exception:
            pass

    if not preds:
        raise RuntimeError("No models available to make predictions")

    # normalize and average
    normalized = []
    for p in preds:
        p = np.asarray(p)
        if n_classes is not None and p.shape[0] != n_classes:
            if p.shape[0] < n_classes:
                pad = np.full((n_classes - p.shape[0],), 1e-8)
                p = np.concatenate([p, pad])
            else:
                p = p[:n_classes]
        if not np.all((p >= 0) & (p <= 1)):
            p = np.exp(p - np.max(p)); p = p / p.sum()
        p = p / p.sum()
        normalized.append(p)

    ensemble = np.mean(np.stack(normalized, axis=0), axis=0)

    if CLASSES:
        confidences = {CLASSES[i]: float(ensemble[i]) for i in range(len(ensemble))}
        idx = int(np.argmax(ensemble))
        return CLASSES[idx], float(ensemble[idx]), confidences
    else:
        confidences = {str(i): float(v) for i,v in enumerate(ensemble)}
        idx = int(np.argmax(ensemble))
        return str(idx), float(ensemble[idx]), confidences
