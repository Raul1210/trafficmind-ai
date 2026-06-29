"""
Detection package.

Re-exports the public surface so callers can write:
    from traffic_monitor.detection import detect_objects, YOLODetector
"""
from traffic_monitor.detection.yolo_detector import YOLODetector, detect_objects

__all__ = ["YOLODetector", "detect_objects"]
