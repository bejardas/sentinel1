from fastapi import FastAPI, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional

from app.database import engine, Base, get_db
from app.models import User, DetectionRecord
from app.schemas import UserCreate, UserOut, Token, DetectionRecordOut
from app.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    get_current_admin_user
)
from app.utils import upload_image_to_cloudinary, generate_google_maps_link
from app.ai_service import ai_engine

# Automatically create database tables in Neon on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Zenith Infrastructure Damage Detection API", version="1.0.0")

# Enable CORS so your frontend teammate can connect easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 1. Authentication Endpoints ---

@app.post("/api/v1/auth/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered.")

    hashed_pwd = get_password_hash(user_data.password)
    new_user = User(username=user_data.username, hashed_password=hashed_pwd, role="standard")

    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/api/v1/auth/login", response_model=Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


# --- 2. Core Detection Endpoints ---

@app.post("/api/v1/detect", response_model=DetectionRecordOut)
async def analyze_infrastructure(
        image: UploadFile = File(...),
        image_category: str = Form(...),
        raw_address: Optional[str] = Form(None),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """
    Uploads an image to Cloudinary, runs YOLOv8 inference, calculates danger level,
    generates map links, and logs the record securely tied to the current user.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

    # 1. Read image buffer
    image_bytes = await image.read()

    # 2. Upload to Cloudinary CDN
    public_id = f"{current_user.username}_{image.filename}_{int(time.time())}" if 'time' in globals() else f"{current_user.username}_{image.filename}"
    # Quick fix for time module import inside function scope if needed
    import time as t
    public_id = f"{current_user.username}_{image.filename}_{int(t.time())}"

    secure_image_url = upload_image_to_cloudinary(image_bytes, public_id)

    # 3. Run AI Inference & Risk Evaluation
    # Since model takes a local path or array, we can save temporarily or pass bytes depending on setup.
    # To keep it robust with YOLO, let's write bytes to a temp local file for inference, then discard.
    temp_path = f"temp_{image.filename}"
    with open(temp_path, "wb") as temp_file:
        temp_file.write(image_bytes)

    try:
        detections, danger_level, execution_time = ai_engine.predict(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 4. Generate Google Maps Link
    map_link = generate_google_maps_link(raw_address) if raw_address else None

    # 5. Calculate metrics
    max_conf = max([d["confidence"] for d in detections]) if detections else 0.0

    # 6. Save to Neon PostgreSQL
    db_record = DetectionRecord(
        filename=image.filename,
        image_category=image_category,
        raw_address=raw_address,
        map_link=map_link,
        detected_classes=detections,
        max_confidence=max_conf,
        danger_level=danger_level,
        image_path=secure_image_url,
        owner_id=current_user.id
    )

    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return db_record


@app.get("/api/v1/detections", response_model=List[DetectionRecordOut])
def get_user_detections(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """
    Standard users see only their uploads. Admins see all records across the system.
    """
    if current_user.role == "admin":
        return db.query(DetectionRecord).all()

    return db.query(DetectionRecord).filter(DetectionRecord.owner_id == current_user.id).all()