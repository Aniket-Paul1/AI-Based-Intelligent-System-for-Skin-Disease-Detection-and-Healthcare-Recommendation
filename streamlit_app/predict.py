import sys
import os

# Absolute project root path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_code.predict_super_ensemble import predict_image


def ensemble_predict(image):
    """
    Calls super ensemble model and formats output for UI.

    Returns:
        label (str)
        confidence (float, percentage)
        top_probs (dict)
    """

    label, confidence, top_probs = predict_image(image)

    # Convert to percentage for Streamlit UI
    confidence_percentage = confidence * 100

    return label, confidence_percentage, top_probs
