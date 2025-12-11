# streamlit_app/app.py
import streamlit as st
from pathlib import Path
from PIL import Image
import os
import json
from .auth import create_user, get_user_by_email, verify_password
from .db import engine
from .models import Base, Prediction, User
from .predict import predict_from_pil, load_models
from .db import SessionLocal
from datetime import datetime

# Ensure DB tables exist for dev (for production use Alembic)
Base.metadata.create_all(bind=engine)

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Load doctor map
with open(Path(__file__).resolve().parent / "doctor_map.json", "r") as f:
    DOCTOR_MAP = json.load(f)

# Streamlit session helpers
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None

st.sidebar.title("Account")
if st.session_state["user_email"]:
    st.sidebar.write(f"Logged in as: {st.session_state['user_email']}")
    if st.sidebar.button("Logout"):
        st.session_state["user_email"] = None
        st.experimental_rerun()
else:
    auth_tab = st.sidebar.radio("Auth", ("Login", "Register"))
    if auth_tab == "Register":
        st.sidebar.subheader("Register")
        reg_email = st.sidebar.text_input("Email", key="reg_email")
        reg_name = st.sidebar.text_input("Full name", key="reg_name")
        reg_pw = st.sidebar.text_input("Password", type="password", key="reg_pw")
        if st.sidebar.button("Create account"):
            existing = get_user_by_email(reg_email)
            if existing:
                st.sidebar.error("Email already registered")
            else:
                create_user(reg_email, reg_pw, reg_name)
                st.sidebar.success("Created. Please login.")
    else:
        st.sidebar.subheader("Login")
        login_email = st.sidebar.text_input("Email", key="login_email")
        login_pw = st.sidebar.text_input("Password", type="password", key="login_pw")
        if st.sidebar.button("Login"):
            user = get_user_by_email(login_email)
            if not user or not verify_password(login_pw, user.hashed_password):
                st.sidebar.error("Invalid credentials")
            else:
                st.session_state["user_email"] = user.email
                st.sidebar.success("Logged in")

st.title("Skin Disease - Prototype (Streamlit)")

st.markdown("Upload a skin photo to get a diagnosis (prototype).")

uploaded = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
if uploaded:
    try:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_column_width=True)
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                # Save file
                ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                filename = f"{st.session_state.get('user_email','anon')}_{ts}_{uploaded.name}"
                save_path = UPLOAD_DIR / filename
                img.save(save_path)

                # Ensure models loaded
                load_models()
                label, conf, confidences = predict_from_pil(img)

                # Map doctor
                rec = DOCTOR_MAP.get(label, {"doctor": "Dermatologist", "urgency": "Normal"})
                doctor = rec["doctor"]
                urgency = rec["urgency"]

                # Save to DB
                db = SessionLocal()
                pred = Prediction(
                    user_id = None,  # could link to user id if logged in
                    image_path = str(save_path),
                    label = label,
                    confidence = str(conf),
                    confidences = confidences,
                    recommended_doctor = doctor,
                    urgency = urgency
                )
                db.add(pred); db.commit(); db.refresh(pred); db.close()

                # Show results
                st.success(f"Diagnosis: {label} ({conf*100:.2f}%)")
                st.write("Recommended doctor:", doctor)
                st.write("Urgency:", urgency)
                st.subheader("Confidence per class")
                for k,v in sorted(confidences.items(), key=lambda x: -x[1])[:10]:
                    st.write(f"{k}: {v*100:.2f}%")
    except Exception as e:
        st.error("Error processing image: " + str(e))
