import os
import time
from typing import List, Dict, Any
from ultralytics import YOLO


class InfrastructureModelService:
    def __init__(self, weights_path: str = "best.pt"):
        self.weights_path = weights_path
        self.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
        self.model = None
        self.load_model()

    def load_model(self):
        """Loads the YOLOv8 model weights, with a fallback stub if weights aren't ready yet."""
        try:
            if os.path.exists(self.weights_path):
                print(f"Loading YOLOv8 weights from {self.weights_path}...")
                self.model = YOLO(self.weights_path)
            else:
                print(f"Warning: {self.weights_path} not found. Operating in MOCK mode until training completes.")
                self.model = None
        except Exception as e:
            print(f"Error loading model weights: {e}. Falling back to mock mode.")
            self.model = None

    def calculate_danger_level(self, detections: List[Dict[str, Any]]) -> str:
        """
        Evaluates risk based on detected damage classes and confidence scores.
        """
        if not detections:
            return "Low"

        # Count high-risk indicators or total count
        total_detections = len(detections)
        max_conf = max([d["confidence"] for d in detections])

        # Example heuristic logic for infrastructure damage risk
        critical_labels = {"exposed_rebar", "severe_spallation", "deep_pothole"}
        has_critical_damage = any(d["label"] in critical_labels for d in detections)

        if has_critical_damage and max_conf > 0.75:
            return "Critical"
        elif total_detections >= 4 or max_conf > 0.85:
            return "High"
        elif total_detections >= 2:
            return "Medium"
        else:
            return "Low"

    def predict(self, image_path: str) -> tuple[List[Dict[str, Any]], str, float]:
        """
        Runs inference on an image and returns detections, calculated danger level, and execution time.
        """
        start_time = time.time()
        detections = []

        if self.model is not None:
            # Real YOLOv8 Inference
            results = self.model(image_path, conf=self.confidence_threshold)
            result = results[0]

            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = result.names.get(class_id, f"class_{class_id}")
                bbox = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]

                detections.append({
                    "class_id": class_id,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "bbox": [round(coord, 2) for coord in bbox]
                })
        else:
            # Fallback mock data if model is still training on Colab
            time.sleep(0.1)  # Simulate inference lag
            detections = [
                {"class_id": 4, "label": "spallation", "confidence": 0.89, "bbox": [100.0, 150.0, 300.0, 400.0]},
                {"class_id": 12, "label": "pothole", "confidence": 0.92, "bbox": [50.0, 80.0, 200.0, 220.0]}
            ]

        danger_level = self.calculate_danger_level(detections)
        execution_time_ms = (time.time() - start_time) * 1000

        return detections, danger_level, round(execution_time_ms, 2)


# Global singleton service instance
ai_engine = InfrastructureModelService()