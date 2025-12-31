import sys
import os
import json
import streamlit as st
from PIL import Image

# --------------------------------------------------
# Fix Python path so Streamlit can see ml_code
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_code.hybrid_predict import hybrid_predict
from db import save_prediction


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="AI Skin Disease Detection",
    page_icon="🧬",
    layout="centered"
)

# --------------------------------------------------
# Title & Description
# --------------------------------------------------
st.title("AI-Based Skin Disease Detection System")

st.markdown(
    "Upload a skin image to obtain an **AI-assisted skin disease prediction**. "
    "The system always predicts the **most visually similar disease** using a "
    "hybrid deep-learning pipeline."
)

# --------------------------------------------------
# Medical Disclaimer
# --------------------------------------------------
st.warning(
    "⚠ **Medical Disclaimer**\n\n"
    "This application is developed for **research and internship evaluation purposes only**. "
    "It is **NOT a medical diagnosis system**. The predictions are based on visual patterns "
    "and similarity analysis. Always consult a certified dermatologist for medical advice."
)

# --------------------------------------------------
# Load Disease → Doctor Mapping
# --------------------------------------------------
@st.cache_data
def load_doctor_map():
    path = os.path.join(os.path.dirname(__file__), "doctor_map.json")
    with open(path, "r") as f:
        return json.load(f)

doctor_map = load_doctor_map()

# --------------------------------------------------
# Image Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a skin image (JPG / PNG / JPEG / WEBP)",
    type=["jpg", "jpeg", "png", "webp"]
)

# --------------------------------------------------
# Display Image
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --------------------------------------------------
    # Analyze Button
    # --------------------------------------------------
    if st.button("Analyze Image"):
        with st.spinner("Analyzing image using AI models..."):
            result = hybrid_predict(image)

        # --------------------------------------------------
        # Extract Result
        # --------------------------------------------------
        label = result["label"]
        confidence = result["confidence"]
        source = result["source"]
        top_probs = result["top_probs"]

        st.success("Analysis completed")

        # --------------------------------------------------
        # Main Prediction Output
        # --------------------------------------------------
        st.markdown(f"### 🧬 Predicted Disease: **{label}**")
        st.markdown(f"**Prediction Source:** `{source}`")
        st.markdown(f"### 📊 Confidence: **{confidence * 100:.2f}%**")

        # --------------------------------------------------
        # Top-2 Similar Diseases
        # --------------------------------------------------
        st.markdown("### 🔍 Most Similar Conditions")

        top2 = list(top_probs.items())[:2]
        for disease, prob in top2:
            st.write(f"- **{disease}**: {prob * 100:.2f}%")

        # --------------------------------------------------
        # Confidence Note (Mentor-Friendly)
        # --------------------------------------------------
        st.info(
            "The prediction is based on **visual similarity and learned patterns**. "
            "Some skin conditions may share overlapping visual characteristics."
        )

        # --------------------------------------------------
        # Doctor Recommendation
        # --------------------------------------------------
        recommended_doctor = doctor_map.get(label, "General Dermatologist")
        st.markdown(
            f"### 👨‍⚕️ Recommended Specialist: **{recommended_doctor}**"
        )

        # --------------------------------------------------
        # Save Prediction History
        # --------------------------------------------------
        save_prediction(
            disease=label,
            confidence=round(confidence * 100, 2)
        )
        st.success("Prediction saved successfully.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "AI Skin Disease Detection System | Infosys Internship Project"
)
