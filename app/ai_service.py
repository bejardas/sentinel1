import time
from ultralytics import YOLO
from fastapi import HTTPException


class AIEngine:
    def __init__(self):
        print("Loading AI Models into memory...")
        # Load both models once when the app boots up
        self.road_model = YOLO("app/weights/road_damage_model.pt")
        self.struct_model = YOLO("app/weights/building_bridge_model.pt")

    def _calculate_road_danger(self, detections):
        if not detections:
            return "Safe / No Road Damage"

        score = 0
        high_risk = ["d40", "d43", "d44"]
        medium_risk = ["d20", "d10", "d11"]

        for d in detections:
            c_name = d["label"].lower()
            if any(k in c_name for k in high_risk):
                score += 10 * d["confidence"]
            elif any(k in c_name for k in medium_risk):
                score += 5 * d["confidence"]
            else:
                score += 2 * d["confidence"]

        if score > 18:
            return "Critical Hazard - Urgent Road Repair Needed"
        elif score > 10:
            return "High Risk - Pavement Deteriorating"
        elif score > 4:
            return "Medium Risk - Monitor Surface"
        return "Low Risk - Minor Surface Blemishes"

    def _calculate_structural_danger(self, detections):
        if not detections:
            return "Safe / No Structural Damage"

        score = 0
        critical_keywords = ["exposedbars", "spallation", "collapse"]
        moderate_keywords = ["efflorescence", "corrosionstain", "crack"]

        for d in detections:
            c_name = d["label"].lower()
            if any(k in c_name for k in critical_keywords):
                score += 8 * d["confidence"]
            elif any(k in c_name for k in moderate_keywords):
                score += 3 * d["confidence"]
            else:
                score += 1

        if score > 20:
            return "Critical - Structural Integrity Compromised"
        elif score > 10:
            return "High Risk - Maintenance Required"
        elif score > 4:
            return "Medium Risk - Monitor Closely"
        return "Low Risk - Minor Surface Wear"

    def predict(self, image_path: str, image_category: str):
        start_time = time.time()

        # Route to the correct model based on the form input
        cat_lower = image_category.lower()
        if "road" in cat_lower or "pothole" in cat_lower:
            primary_model = self.road_model
            fallback_model = self.struct_model
            is_road = True
        else:
            primary_model = self.struct_model
            fallback_model = self.road_model
            is_road = False

        # Run primary model
        results = primary_model(image_path)
        active_model = primary_model

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = active_model.names[cls_id]

                if cls_name.lower() != "background":
                    detections.append({
                        "class_id": cls_id,
                        "label": cls_name,
                        "confidence": round(float(box.conf[0]), 2),
                        "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()]
                    })

        # EDGE CASE FIX 1: If user selected wrong category and 0 detections found, try fallback model
        if not detections:
            print("Primary model found nothing. Triggering cross-category fallback...")
            active_model = fallback_model
            is_road = not is_road
            results = active_model(image_path)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = active_model.names[cls_id]

                    if cls_name.lower() != "background":
                        detections.append({
                            "class_id": cls_id,
                            "label": cls_name,
                            "confidence": round(float(box.conf[0]), 2),
                            "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()]
                        })

        # EDGE CASE FIX 2: Relaxed threshold (0.20) to ensure legit photos with lower confidence pass through
        if not detections or max([d["confidence"] for d in detections]) < 0.20:
            raise HTTPException(
                status_code=400,
                detail="Invalid input: Image does not appear to contain valid road or structural damage."
            )

        # Calculate danger level tailored to the category
        if is_road:
            danger_level = self._calculate_road_danger(detections)
        else:
            danger_level = self._calculate_structural_danger(detections)

        execution_time_ms = round((time.time() - start_time) * 1000, 2)

        return detections, danger_level, execution_time_ms


# Global instance imported by main.py
ai_engine = AIEngine()