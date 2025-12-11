# streamlit_app/predict.py
"""
Streamlit-side prediction adapter — imports ml_code.predict/predict_super_ensemble
and exposes a simple function call for the app.
"""
from PIL import Image
import traceback

# Prefer the higher-level ensemble if available
try:
    from ml_code.predict_super_ensemble import predict_from_pil as ensemble_predict
except Exception:
    ensemble_predict = None

try:
    from ml_code.predict import predict_from_pil as simple_predict
except Exception:
    simple_predict = None

def predict_image_pil(img_pil):
    """
    Returns: (label, confidence, confidences_dict)
    Tries ensemble first, then simple predictor.
    """
    try:
        if ensemble_predict is not None:
            return ensemble_predict(img_pil)["label"], ensemble_predict(img_pil)["confidence"], ensemble_predict(img_pil)["confidences"]
    except Exception as e:
        # If ensemble fails, fallback to simple_predict
        print("Ensemble predict failed:", e)
        traceback.print_exc()

    try:
        if simple_predict is not None:
            return simple_predict(img_pil)
    except Exception as e:
        print("Simple predict failed:", e)
        traceback.print_exc()
    raise RuntimeError("No prediction function available or all predictors failed.")
