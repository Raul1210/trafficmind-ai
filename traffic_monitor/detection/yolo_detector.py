"""
YOLOv8 object detection backend.

Design goals:
  - Swappable: add a new detector by subclassing BaseDetector and updating the
    factory at the bottom — zero changes to the rest of the codebase.
  - Singleton: one model instance shared across the application; YOLO inference
    is read-only on the loaded weights so this is thread-safe when reads are
    serialised by the capture worker's internal lock.
  - Fail-fast: FileNotFoundError is raised (and caught by create_app's
    validate_paths call) before any thread starts.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from traffic_monitor.config import settings
from traffic_monitor.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------
DetectionResult = tuple[
    Optional[np.ndarray],   # frame (may be None on error)
    dict[str, int],         # counts: {"car": N, "motorbike": N, "person": N}
    list[tuple],            # boxes: [(x1, y1, x2, y2, label, conf), ...]
    float,                  # avg_confidence over accepted detections
]


# ---------------------------------------------------------------------------
# Detector class (extend this to swap in a different model)
# ---------------------------------------------------------------------------

class YOLODetector:
    """
    Wraps an ultralytics YOLO model.

    To add a new backend (e.g. TensorRT, ONNX):
        1. Subclass YOLODetector
        2. Override detect()
        3. Update _build_detector() below
    """

    def __init__(self, model_path: Path, inference_size: int) -> None:
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO model file not found: {model_path}\n"
                f"  Set YOLO_MODEL_PATH env var or place the file at that path."
            )

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(model_path), task='detect')
            logger.info(f"YOLO model loaded: {model_path.name}")
        except ImportError:
            msg = "ultralytics not installed. Run: pip install ultralytics"
            logger.critical(msg)
            sys.exit(msg)

        self._inference_size = inference_size

    # ── Public interface ──────────────────────────────────────────────────

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
        class_mapping: Optional[dict] = None,
        class_min_confidence: Optional[dict] = None,
    ) -> DetectionResult:
        """
        Run inference on a single BGR frame.

        Args:
            frame: numpy array (H, W, 3) in BGR colour space.
            confidence_threshold: override the global detection threshold.
            class_mapping: override label consolidation map.
            class_min_confidence: override per-class min confidence.

        Returns:
            DetectionResult tuple (frame, counts, boxes, avg_confidence).
        """
        if frame is None or not isinstance(frame, np.ndarray):
            logger.warning("detect: received invalid or None frame")
            return None, {"car": 0, "motorbike": 0, "person": 0}, [], 0.0

        threshold  = confidence_threshold  or settings.detection_confidence_threshold
        cls_map    = class_mapping         or settings.class_mapping
        cls_min    = class_min_confidence  or settings.class_min_confidence

        counts = {"car": 0, "motorbike": 0, "person": 0}
        boxes: list[tuple] = []
        total_conf = 0.0
        num_det = 0

        try:
            results = self._model(
                frame,
                stream=False,
                verbose=False,
                conf=threshold,
                imgsz=self._inference_size,
            )
        except Exception as e:
            logger.error(f"YOLO inference error: {e}")
            return frame, counts, boxes, 0.0

        if not results:
            return frame, counts, boxes, 0.0

        for r in results:
            if not hasattr(r, "boxes") or r.boxes is None:
                continue
            for box in r.boxes:
                try:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])

                    if not (0 <= conf <= 1):
                        continue

                    raw_label = self._model.names.get(cls_id, "unknown")
                    if raw_label not in cls_map:
                        continue

                    final_label = cls_map[raw_label]
                    if conf < cls_min.get(final_label, threshold):
                        continue

                    b = box.xyxy[0].cpu().numpy().astype(int)
                    counts[final_label] += 1
                    boxes.append((*b, final_label, conf))
                    total_conf += conf
                    num_det += 1

                except Exception as e:
                    logger.debug(f"Box processing error: {e}")

        avg_conf = (total_conf / num_det) if num_det > 0 else 0.0
        if num_det > 0:
            logger.debug(
                f"Detected: cars={counts['car']} bikes={counts['motorbike']} "
                f"persons={counts['person']} avg_conf={avg_conf:.2f}"
            )
        return frame, counts, boxes, avg_conf


# ---------------------------------------------------------------------------
# FluidDetector
# Frame-skipping wrapper: runs YOLO every (skip_frames+1) calls;
# returns the cached annotated frame on skipped calls for smooth display.
#
# Also implements IoU-based box persistence (TASK 4) to suppress flickering:
# a detection slot is kept for up to _MAX_MISSING frames after YOLO stops
# reporting it, matched to new detections by bounding-box IoU.
# ---------------------------------------------------------------------------

class FluidDetector:
    """
    Wraps a raw ultralytics YOLO model with frame-skip + anti-flicker logic.

    process(frame) returns (annotated_frame, counts, boxes, avg_conf).

    skip_frames=0  → inference on every frame
    skip_frames=1  → inference on every 2nd frame
    skip_frames=2  → inference on every 3rd frame  (default for yolov8s on CPU)
    """

    # COCO class IDs — vehicles only; person excluded to reduce inference cost.
    # Pedestrian count in the dashboard will read 0 when using these classes.
    _VEHICLE_CLASSES = [0, 2, 3, 5, 7]   # car, motorcycle, bus, truck

    # Dashboard colour scheme (BGR) for custom box drawing
    _COLORS = {
        "car":       (255, 229,   0),   # cyan   #00e5ff
        "motorbike": (  0, 179, 255),   # amber  #ffb300
        "person":    (255, 100, 180),   # purple #b464ff
    }
    _COLOR_DEFAULT = (200, 200, 200)

    def __init__(self, model, skip_frames: int = 2, inference_size: int = 480) -> None:
        self._model          = model          # raw ultralytics YOLO instance
        self._skip_frames    = skip_frames
        self._inference_size = inference_size
        self._frame_count    = 0
        self._lock           = threading.Lock()

        # Cached state — returned on skipped frames
        self._last_annotated: Optional[np.ndarray] = None
        self._last_counts:    dict[str, int]        = {"car": 0, "motorbike": 0, "person": 0}
        self._last_boxes:     list[tuple]           = []
        self._last_conf:      float                 = 0.0

        # ── Anti-flicker persistence (TASK 4) ────────────────────────────
        # Each slot: {"coord": (x1,y1,x2,y2), "label": str, "conf": float,
        #             "missing": int}   # frames since last matched detection
        self._persistent:   list[dict] = []
        self._MAX_MISSING = 3      # keep a ghost box for up to 3 frames
        self._IOU_THRESH  = 0.25   # minimum IoU to match a new detection to a slot

        self._cls_map = settings.class_mapping
        self._cls_min = settings.class_min_confidence

    # ── Public ────────────────────────────────────────────────────────────

    def reset_state(self, new_inference_size: Optional[int] = None) -> None:
        """
        Clear all cached state and optionally change inference size.

        Must be called after a video source switch so ghost boxes from the
        previous source don't bleed into the first frames of the new source.
        """
        with self._lock:
            self._frame_count    = 0
            self._last_annotated = None
            self._last_counts    = {"car": 0, "motorbike": 0, "person": 0}
            self._last_boxes     = []
            self._last_conf      = 0.0
            self._persistent     = []
            if new_inference_size is not None and new_inference_size > 0:
                self._inference_size = new_inference_size
                logger.info(f"FluidDetector: inference_size → {new_inference_size}")

    def process(
        self, frame: np.ndarray
    ) -> tuple[Optional[np.ndarray], dict[str, int], list[tuple], float]:
        """
        Run inference or return cached result.

        Returns:
            (annotated_frame, counts, boxes, avg_conf)
        """
        if frame is None or not isinstance(frame, np.ndarray):
            return None, {"car": 0, "motorbike": 0, "person": 0}, [], 0.0

        self._frame_count += 1

        # ── Skipped frame: return cached result ───────────────────────────
        if self._frame_count % (self._skip_frames + 1) != 0:
            with self._lock:
                if self._last_annotated is not None:
                    return (
                        self._last_annotated,
                        dict(self._last_counts),
                        list(self._last_boxes),
                        self._last_conf,
                    )
            return frame, {"car": 0, "motorbike": 0, "person": 0}, [], 0.0

        # ── Inference frame ───────────────────────────────────────────────
        try:
            results = self._model(
                frame,
                imgsz=self._inference_size,
                conf=0.15,          # 0.15 catches small/distant vehicles missed at 0.25
                iou=0.40,           # lower → fewer duplicate boxes   (TASK 3)
                classes=self._VEHICLE_CLASSES,
                verbose=False,
                stream=False,
                half=False,         # CPU does not support FP16
            )
            result = results[0]
        except Exception as e:
            logger.error(f"FluidDetector inference error: {e}")
            with self._lock:
                return (
                    self._last_annotated or frame,
                    dict(self._last_counts),
                    list(self._last_boxes),
                    self._last_conf,
                )

        raw_counts, raw_boxes, avg_conf = self._extract_stats(result)

        # Apply persistence filter — suppresses flicker (TASK 4)
        smooth_boxes = self._apply_smoothing(raw_boxes)

        # Recount from smoothed boxes (persistent ghosts keep the count stable)
        counts: dict[str, int] = {"car": 0, "motorbike": 0, "person": 0}
        for box in smooth_boxes:
            lbl = box[4]
            counts[lbl] = counts.get(lbl, 0) + 1

        # Draw smoothed boxes on raw frame (custom style matching dashboard)
        annotated = self._draw_boxes(frame, smooth_boxes)

        with self._lock:
            self._last_annotated = annotated
            self._last_counts    = counts
            self._last_boxes     = smooth_boxes
            self._last_conf      = avg_conf

        return annotated, counts, smooth_boxes, avg_conf

    # ── Private — inference helpers ───────────────────────────────────────

    def _extract_stats(
        self, result
    ) -> tuple[dict[str, int], list[tuple], float]:
        """Extract counts / boxes / avg_conf from an ultralytics Result."""
        counts: dict[str, int] = {"car": 0, "motorbike": 0, "person": 0}
        boxes:  list[tuple]    = []
        total_conf = 0.0
        num_det    = 0

        if not hasattr(result, "boxes") or result.boxes is None:
            return counts, boxes, 0.0

        for box in result.boxes:
            try:
                cls_id      = int(box.cls[0])
                conf        = float(box.conf[0])
                raw_label   = self._model.names.get(cls_id, "unknown")
                if raw_label not in self._cls_map:
                    continue
                final_label = self._cls_map[raw_label]
                # Apply per-class minimum confidence (mirrors YOLODetector.detect)
                if conf < self._cls_min.get(final_label, 0.15):
                    continue
                b = box.xyxy[0].cpu().numpy().astype(int)
                counts[final_label] = counts.get(final_label, 0) + 1
                boxes.append((*b, final_label, conf))
                total_conf += conf
                num_det    += 1
            except Exception:
                pass

        avg_conf = total_conf / num_det if num_det > 0 else 0.0
        return counts, boxes, avg_conf

    # ── Private — anti-flicker (TASK 4) ──────────────────────────────────

    @staticmethod
    def _box_iou(a: tuple, b: tuple) -> float:
        """Compute IoU between two (x1, y1, x2, y2) coordinate tuples."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1);  iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2);  iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def _apply_smoothing(self, new_boxes: list[tuple]) -> list[tuple]:
        """
        Persist detection slots across frames to suppress one-frame dropouts.

        Algorithm:
          1. Each new box is matched to the closest existing slot (by IoU).
          2. Matched slots are updated with the fresh coordinates / confidence.
          3. Unmatched slots have their 'missing' counter incremented.
          4. Slots with missing > _MAX_MISSING are evicted.
          5. New boxes with no matching slot create a fresh slot.

        Returns a flat list of (x1,y1,x2,y2,label,conf) tuples covering
        both confirmed and ghost detections.
        """
        matched_idx: set[int] = set()

        for box in new_boxes:
            x1, y1, x2, y2, label, conf = box[:6]
            new_coord = (int(x1), int(y1), int(x2), int(y2))

            best_iou, best_i = 0.0, -1
            for i, slot in enumerate(self._persistent):
                if slot["label"] != label:
                    continue
                iou = self._box_iou(new_coord, slot["coord"])
                if iou > best_iou:
                    best_iou, best_i = iou, i

            if best_i >= 0 and best_iou >= self._IOU_THRESH:
                # Refresh existing slot
                self._persistent[best_i]["coord"]   = new_coord
                self._persistent[best_i]["conf"]    = float(conf)
                self._persistent[best_i]["missing"] = 0
                matched_idx.add(best_i)
            else:
                # Brand-new detection
                self._persistent.append(
                    {"coord": new_coord, "label": label,
                     "conf": float(conf), "missing": 0}
                )
                matched_idx.add(len(self._persistent) - 1)

        # Age unmatched slots; evict stale ones
        survivors: list[dict] = []
        for i, slot in enumerate(self._persistent):
            if i not in matched_idx:
                slot["missing"] += 1
            if slot["missing"] <= self._MAX_MISSING:
                survivors.append(slot)
        self._persistent = survivors

        return [
            (*s["coord"], s["label"], s["conf"])
            for s in self._persistent
        ]

    # ── Private — drawing (TASK 4) ────────────────────────────────────────

    def _draw_boxes(
        self, frame: np.ndarray, boxes: list[tuple]
    ) -> np.ndarray:
        """
        Draw smoothed bounding boxes using the dashboard colour scheme.
        Replaces result.plot() so ghost/persistent boxes are included.
        """
        import cv2  # local import avoids circular dep at module load time
        annotated = frame.copy()

        for box in boxes:
            if len(box) < 6:
                continue
            x1, y1, x2, y2, label, conf = box[:6]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = self._COLORS.get(label, self._COLOR_DEFAULT)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            text = f"{label} {float(conf):.0%}"
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
            )
            label_y = max(y1 - 4, th + 6)
            cv2.rectangle(
                annotated,
                (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2),
                color, cv2.FILLED,
            )
            cv2.putText(
                annotated, text, (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
            )

        return annotated


# ---------------------------------------------------------------------------
# Factory — swap the backend by changing this function only
# ---------------------------------------------------------------------------

def _build_detector() -> YOLODetector:
    """
    Prefer yolov8s.onnx (small, fast CPU) when present; fall back to yolov8m.pt.
    Run export_onnx.py once to generate the ONNX file.
    """
    small_onnx = settings.yolo_small_onnx_path
    if small_onnx.exists():
        logger.info(f"Loading optimised ONNX model: {small_onnx.name}")
        try:
            return YOLODetector(
                model_path=small_onnx,
                inference_size=settings.yolo_inference_size,
            )
        except Exception as e:
            logger.warning(f"ONNX load failed ({e}), falling back to .pt model")

    logger.info(
        f"ONNX not found at {small_onnx} — using {settings.yolo_model_path.name}"
    )
    try:
        return YOLODetector(
            model_path=settings.yolo_model_path,
            inference_size=settings.yolo_inference_size,
        )
    except FileNotFoundError as e:
        logger.critical(str(e))
        sys.exit(1)


# Module-level singletons — created once when the module is first imported.
_detector: YOLODetector = _build_detector()

# FluidDetector: skip_frames=2 → inference every 3rd frame for yolov8s on CPU.
# Tune down to skip_frames=1 if FPS is consistently above 20.
fluid_detector: FluidDetector = FluidDetector(
    model=_detector._model,
    skip_frames=2,
    inference_size=settings.yolo_inference_size,
)


# ---------------------------------------------------------------------------
# Public shim — identical signature to the original detect_objects() so
# existing callers (and the old object_detection.py) need no changes.
# ---------------------------------------------------------------------------

def detect_objects(
    frame: np.ndarray,
    confidence_threshold: Optional[float] = None,
) -> DetectionResult:
    """Thin wrapper around the module-level YOLODetector singleton."""
    return _detector.detect(frame, confidence_threshold=confidence_threshold)
