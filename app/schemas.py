from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

# --- Authentication & User Schemas ---
class UserCreate(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    role: str

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str


# --- Detection & AI Response Schemas ---
class DetectionItem(BaseModel):
    class_id: int
    label: str
    confidence: float
    bbox: List[float]  # [ymin, xmin, ymax, xmax]

# The complete payload returned to your frontend developer
class DetectionRecordOut(BaseModel):
    id: int
    filename: str
    image_category: str
    raw_address: Optional[str] = None
    map_link: Optional[str] = None
    danger_level: str
    total_detections: int
    detections: List[DetectionItem]
    execution_time_ms: float
    image_path: str  # Cloudinary secure URL
    created_at: datetime
    owner_id: int

    class Config:
        from_attributes = True