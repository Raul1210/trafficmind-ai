# Object Detection Script - YOLOv8m Engine (Optimized 640px Inference)
import cv2
import numpy as np
import os
import logging
import sys

# Import configuration
from config import (
    YOLO_MODEL_PATH, DETECTION_CONFIDENCE_THRESHOLD, YOLO_INFERENCE_SIZE,
    CLASS_MAPPING, CLASS_MIN_CONFIDENCE, LOG_LEVEL, LOG_FORMAT
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Use the model path defined in config (avoids hardcoded path mismatch)
ACTIVE_MODEL = YOLO_MODEL_PATH

model = None
yolo_error_message = None

try:
    from pathlib import Path
    from ultralytics import YOLO

    if not Path(ACTIVE_MODEL).exists():
        raise FileNotFoundError(f"YOLO model file not found: {ACTIVE_MODEL}")

    model = YOLO(ACTIVE_MODEL)
    logger.info(f"YOLO model loaded successfully: {ACTIVE_MODEL}")

except ImportError as e:
    yolo_error_message = "ultralytics library not found. Run: pip install ultralytics"
    logger.critical(yolo_error_message)
    sys.exit(yolo_error_message)

except FileNotFoundError as e:
    yolo_error_message = str(e)
    logger.critical(yolo_error_message)
    sys.exit(yolo_error_message)

except Exception as e:
    yolo_error_message = f"CRITICAL ERROR loading YOLO model: {type(e).__name__}: {str(e)}"
    logger.critical(yolo_error_message)
    sys.exit(yolo_error_message)


def detect_objects(frame, confidence_threshold=None):
    """
    Run YOLOv8m inference. Returns boxes with confidence scores.

    Args:
        frame: Input image frame (numpy array)
        confidence_threshold: Minimum confidence (0-1). None = use config default.

    Returns:
        Tuple of (frame, counts_dict, boxes_list, avg_confidence)
    """
    if confidence_threshold is None:
        confidence_threshold = DETECTION_CONFIDENCE_THRESHOLD

    try:
        if frame is None:
            logger.warning("detect_objects: Received None frame")
            return None, {}, [], 0.0

        if not isinstance(frame, np.ndarray):
            logger.error(f"detect_objects: Invalid frame type {type(frame)}")
            return None, {}, [], 0.0

        counts = {"car": 0, "motorbike": 0, "person": 0}
        final_boxes = []
        total_conf = 0.0
        num_detections = 0

        if model is None:
            logger.error("detect_objects: YOLO model not loaded.")
            return frame, counts, final_boxes, 0.0

        try:
            results = model(
                frame,
                stream=False,
                verbose=False,
                conf=confidence_threshold,
                imgsz=YOLO_INFERENCE_SIZE
            )

            if results is None or len(results) == 0:
                return frame, counts, final_boxes, 0.0

            for r in results:
                if not hasattr(r, 'boxes') or r.boxes is None:
                    continue

                for box in r.boxes:
                    try:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])

                        if conf < 0 or conf > 1:
                            continue

                        label = model.names.get(cls_id, "unknown")

                        if label in CLASS_MAPPING:
                            final_label = CLASS_MAPPING[label]
                            if conf < CLASS_MIN_CONFIDENCE.get(final_label, confidence_threshold):
                                continue

                            counts[final_label] += 1
                            b = box.xyxy[0].cpu().numpy().astype(int)
                            final_boxes.append((*b, final_label, conf))
                            total_conf += conf
                            num_detections += 1

                    except Exception as box_error:
                        logger.debug(f"Error processing box: {box_error}")
                        continue

        except RuntimeError as runtime_error:
            logger.error(f"detect_objects: YOLO inference error: {runtime_error}")
            return frame, counts, final_boxes, 0.0
        except Exception as inference_error:
            logger.error(f"detect_objects: Unexpected inference error: {inference_error}")
            return frame, counts, final_boxes, 0.0

        avg_confidence = (total_conf / num_detections) if num_detections > 0 else 0.0

        if num_detections > 0:
            logger.debug(
                f"Detections: cars={counts['car']}, bikes={counts['motorbike']}, "
                f"persons={counts['person']}, avg_conf={avg_confidence:.2f}"
            )

        return frame, counts, final_boxes, avg_confidence

    except Exception as e:
        logger.error(f"detect_objects: Critical error: {type(e).__name__}: {e}")
        return frame, {"car": 0, "motorbike": 0, "person": 0}, [], 0.0