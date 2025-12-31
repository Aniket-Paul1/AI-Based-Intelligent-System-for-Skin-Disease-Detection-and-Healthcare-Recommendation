from PIL import Image

from ml_code.predict_super_ensemble import predict_from_pil
from ml_code.clip_zero_shot import clip_zero_shot_predict

CNN_CONF_THRESHOLD = 0.30  # Lowered for web images


def hybrid_predict(image: Image.Image):
    """
    Always predicts an exact disease name.
    Uses hierarchy to prevent catastrophic errors.
    """

    # ---------------- CNN FIRST ----------------
    cnn_result = predict_from_pil(image)
    cnn_label = cnn_result["label"]
    cnn_conf = cnn_result["confidence"]
    cnn_probs = cnn_result["confidences"]

    # ---------------- CANCER PRIORITY ----------------
    # If CNN even moderately believes cancer → force it
    if cnn_label in ["Melanoma", "Basal Cell Carcinoma"] and cnn_conf >= 0.25:
        return {
            "label": cnn_label,
            "confidence": cnn_conf,
            "source": "CNN (Cancer Priority)",
            "top_probs": cnn_probs,
        }

    # ---------------- CNN NORMAL ----------------
    if cnn_conf >= CNN_CONF_THRESHOLD:
        return {
            "label": cnn_label,
            "confidence": cnn_conf,
            "source": "CNN (Trained Disease)",
            "top_probs": cnn_probs,
        }

    # ---------------- CLIP SIMILARITY FALLBACK ----------------
    zs_label, zs_conf, zs_probs = clip_zero_shot_predict(image)

    return {
        "label": zs_label,
        "confidence": zs_conf,
        "source": "CLIP Similarity (Fallback)",
        "top_probs": zs_probs,
    }
