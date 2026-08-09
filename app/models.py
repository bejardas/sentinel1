from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime, timezone
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="standard", nullable=False)  # "standard" or "admin"

    # Relationship to user detections
    detections = relationship("DetectionRecord", back_populates="owner")


class DetectionRecord(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    image_category = Column(String, index=True, nullable=False)  # Sent by frontend

    # Location Metadata
    raw_address = Column(String, nullable=True)
    map_link = Column(String, nullable=True)

    # AI Analysis & Risk Metadata
    detected_classes = Column(JSONB, nullable=False)
    max_confidence = Column(Float, nullable=False)
    danger_level = Column(String, nullable=False)  # "Low", "Medium", "High", "Critical"

    image_path = Column(String, nullable=False)  # Cloudinary URL
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Cryptographic/User Ownership Isolation
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    owner = relationship("User", back_populates="detections")