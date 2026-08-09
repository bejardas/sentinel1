import os
import urllib.parse
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

# Configure Cloudinary using credentials from .env
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def upload_image_to_cloudinary(file_bytes: bytes, public_id: str) -> str:
    """
    Uploads raw image bytes to Cloudinary and returns the secure HTTPS CDN URL.
    """
    try:
        # Upload buffer directly to Cloudinary
        upload_result = cloudinary.uploader.upload(
            file_bytes,
            public_id=public_id,
            folder="zenith_infrastructure"
        )
        return upload_result.get("secure_url")
    except Exception as e:
        raise RuntimeError(f"Cloudinary upload failed: {str(e)}")

def generate_google_maps_link(raw_address: str) -> str:
    """
    Converts a plain text address into a clickable Google Maps query URL.
    """
    if not raw_address:
        return ""
    encoded_address = urllib.parse.quote(raw_address.strip())
    return f"https://www.google.com/maps/search/?api=1&query={encoded_address}"