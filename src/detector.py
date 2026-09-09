"""
SHIPGUARD AI - Real Computer Vision Defect Detector
Performs actual inference using trained YOLOv8 weights on ship hull inspection imagery.
Strictly conforms to the No Fake Data Policy.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


# Non-structural classes present in upstream training sets that must be filtered out
NON_STRUCTURAL_CLASSES = {"car", "automobile", "vehicle", "person"}

# High-visibility color palette for maritime defects (BGR format for OpenCV)
CLASS_COLORS = {
    "severe-corrosion": (0, 0, 230),     # Vivid Crimson Red
    "corrosion": (0, 102, 255),          # Safety Orange
    "moderate-corrosion": (0, 165, 255), # Amber Orange
    "mild-corrosion": (0, 215, 255),     # Golden Yellow
    "iron rust": (42, 42, 165),          # Deep Rust Red
    "copper corrosion": (128, 128, 0),   # Marine Verdigris Teal
    "rust": (0, 140, 255),               # Dark Orange
    "corroded-part": (30, 30, 200),      # Brick Red
    "crack": (255, 0, 255),              # Magenta / Violet
    "default": (0, 220, 255)             # Bright Cyan / Yellow
}


class ShipGuardDetector:
    """
    Object detection engine for ship hull visual inspection.
    Loads real YOLO weights and returns structured detection objects.
    """

    def __init__(
        self,
        model_path: str = "model/best.pt",
        crack_model_path: Optional[str] = "model/crack_best.pt",
        default_conf: float = 0.15,
        default_iou: float = 0.45
    ):
        self.model_path = model_path
        self.crack_model_path = crack_model_path
        self.default_conf = default_conf
        self.default_iou = default_iou
        self.model = None
        self.crack_model = None
        self.is_loaded = False
        self.load_error = None
        
        self._load_models()

    def _load_models(self):
        """Loads YOLO weights into memory."""
        if not ULTRALYTICS_AVAILABLE:
            self.load_error = "Ultralytics package is not installed."
            return

        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                self.is_loaded = True
            else:
                self.load_error = f"Primary model weights not found at {self.model_path}"

            if self.crack_model_path and os.path.exists(self.crack_model_path):
                self.crack_model = YOLO(self.crack_model_path)
        except Exception as e:
            self.load_error = f"Failed to load model weights: {str(e)}"
            self.is_loaded = False

    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about the loaded models."""
        info = {
            "primary_model_loaded": self.model is not None,
            "primary_model_path": self.model_path,
            "crack_model_loaded": self.crack_model is not None,
            "crack_model_path": self.crack_model_path,
            "classes": {},
            "status": "Ready" if self.is_loaded else "Unavailable",
            "error": self.load_error
        }
        if self.model and hasattr(self.model, "names"):
            info["classes"]["primary"] = [
                name for name in self.model.names.values() 
                if name.lower() not in NON_STRUCTURAL_CLASSES
            ]
        if self.crack_model and hasattr(self.crack_model, "names"):
            info["classes"]["crack"] = list(self.crack_model.names.values())
        return info

    def detect(
        self,
        image_input: Any,
        conf_threshold: Optional[float] = None,
        enable_crack_detector: bool = True
    ) -> Dict[str, Any]:
        """
        Executes actual model inference on an image.

        Args:
            image_input: File path (str), PIL.Image, or numpy.ndarray (RGB)
            conf_threshold: Custom confidence threshold (e.g. 0.15)
            enable_crack_detector: Whether to also run crack model if available

        Returns:
            Dictionary containing raw detections, statistics, and image metadata.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"Detector is not operational: {self.load_error or 'Unknown initialization error'}")

        conf = conf_threshold if conf_threshold is not None else self.default_conf

        # Load and normalize image to RGB NumPy array
        np_image, original_dims = self._preprocess_image(image_input)
        img_h, img_w = original_dims
        total_image_area = img_h * img_w

        detections: List[Dict[str, Any]] = []

        # 1. Run primary model (corrosion & surface defects)
        results_primary = self.model.predict(np_image, conf=conf, iou=self.default_iou, verbose=False)
        for box in results_primary[0].boxes:
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names[cls_id]

            # Filter out non-structural classes
            if cls_name.lower() in NON_STRUCTURAL_CLASSES:
                continue

            confidence = float(box.conf[0].item())
            xyxy = [float(coord) for coord in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = xyxy
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            box_area = w * h
            rel_area = (box_area / total_image_area) if total_image_area > 0 else 0.0

            detections.append({
                "class_name": cls_name,
                "confidence": round(confidence, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "box_width": round(w, 1),
                "box_height": round(h, 1),
                "relative_area_pct": round(rel_area * 100.0, 3),
                "model_source": "primary_corrosion"
            })

        # 2. Run secondary crack model if enabled and available
        if enable_crack_detector and self.crack_model is not None:
            results_crack = self.crack_model.predict(np_image, conf=conf, iou=self.default_iou, verbose=False)
            for box in results_crack[0].boxes:
                cls_id = int(box.cls[0].item())
                cls_name = self.crack_model.names[cls_id]
                confidence = float(box.conf[0].item())
                xyxy = [float(coord) for coord in box.xyxy[0].tolist()]
                x1, y1, x2, y2 = xyxy
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                box_area = w * h
                rel_area = (box_area / total_image_area) if total_image_area > 0 else 0.0

                detections.append({
                    "class_name": cls_name,
                    "confidence": round(confidence, 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "box_width": round(w, 1),
                    "box_height": round(h, 1),
                    "relative_area_pct": round(rel_area * 100.0, 3),
                    "model_source": "crack_detection"
                })

        # Sort detections by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # Generate summary metrics
        counts_by_class: Dict[str, int] = {}
        total_rel_area = 0.0
        for d in detections:
            c_name = d["class_name"]
            counts_by_class[c_name] = counts_by_class.get(c_name, 0) + 1
            total_rel_area += d["relative_area_pct"]

        highest_conf = detections[0]["confidence"] if detections else 0.0
        avg_conf = (sum(d["confidence"] for d in detections) / len(detections)) if detections else 0.0

        # Generate annotated image
        annotated_image = self.draw_annotations(np_image, detections)

        return {
            "detections": detections,
            "total_defects": len(detections),
            "class_counts": counts_by_class,
            "highest_confidence": round(highest_conf, 4),
            "average_confidence": round(avg_conf, 4),
            "total_defect_area_pct": round(min(100.0, total_rel_area), 2),
            "image_dimensions": {"width": img_w, "height": img_h},
            "annotated_image": annotated_image,
            "original_image": np_image
        }

    def draw_annotations(self, image_rgb: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Renders clear, professional bounding boxes and callout labels on the image.
        Uses high-contrast naval maritime theme colors.
        """
        # Work on a copy in BGR format for OpenCV drawing
        img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_name = det["class_name"]
            conf = det["confidence"]

            # Select color based on class
            color = CLASS_COLORS.get(cls_name.lower(), CLASS_COLORS["default"])

            # 1. Bounding box with rounded look or corner brackets
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # 2. Semi-transparent fill inside the defect region
            overlay = img_bgr.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.12, img_bgr, 0.88, 0, img_bgr)

            # 3. Label tag formatting
            label = f"{cls_name.upper()} {int(conf * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Position badge above or inside box if at top edge
            badge_y1 = max(0, y1 - text_h - 8)
            badge_y2 = badge_y1 + text_h + 8
            badge_x2 = min(img_bgr.shape[1], x1 + text_w + 10)

            # Draw badge background
            cv2.rectangle(img_bgr, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

            # Draw white text
            cv2.putText(
                img_bgr,
                label,
                (x1 + 5, badge_y2 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA
            )

        # Convert back to RGB
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _preprocess_image(self, image_input: Any) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Normalizes input image to RGB NumPy array and returns its (height, width)."""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Inspection image file not found: {image_input}")
            pil_img = Image.open(image_input)
            pil_img = pil_img.convert("RGB")
            np_img = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            np_img = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                np_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                np_img = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                np_img = image_input.copy()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        return np_img, (np_img.shape[0], np_img.shape[1])
