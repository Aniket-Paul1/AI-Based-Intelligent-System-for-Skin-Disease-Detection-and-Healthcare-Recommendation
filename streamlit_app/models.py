# streamlit_app/models.py
from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func
from .db import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)      # optional link to user
    image_path = Column(String, nullable=False)   # saved path to upload
    label = Column(String, nullable=False)
    confidence = Column(String, nullable=False)   # e.g., "0.85"
    confidences = Column(JSON, nullable=True)     # per-class confidences as JSON
    recommended_doctor = Column(String, nullable=True)
    urgency = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
