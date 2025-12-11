# streamlit_app/auth.py
from passlib.context import CryptContext
from sqlalchemy.future import select
from .models import User
from .db import SessionLocal

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

def create_user(email: str, password: str, full_name: str = None):
    db = SessionLocal()
    try:
        user = User(email=email, full_name=full_name, hashed_password=hash_password(password))
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    finally:
        db.close()

def get_user_by_email(email: str):
    db = SessionLocal()
    try:
        stmt = select(User).where(User.email == email)
        res = db.execute(stmt).scalars().first()
        return res
    finally:
        db.close()
