"""
Centralized settings for Smart Traffic Management System.

All tuneable values are loaded from environment variables with safe defaults.
Import the module-level `settings` singleton everywhere — never instantiate
Settings directly.

Usage:
    from traffic_monitor.config import settings, validate_paths
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Directory anchors (resolved at import time — never changes at runtime)
# ---------------------------------------------------------------------------
_PKG_DIR: Path = Path(__file__).parent          # traffic_monitor/
_PROJECT_DIR: Path = _PKG_DIR.parent            # project root (where run.py, model files, etc. all live)


class Settings:
    """
    Immutable-by-convention configuration object.
    All values loaded from environment variables; defaults mirror the
    original config.py so the app behaves identically out of the box.
    """

    def __init__(self) -> None:
        # ── Directory layout ──────────────────────────────────────────────
        self.base_dir: Path = _PKG_DIR
        self.project_dir: Path = _PROJECT_DIR

        # ── File paths ────────────────────────────────────────────────────
        self.yolo_model_path: Path = Path(
            os.getenv("YOLO_MODEL_PATH", str(_PROJECT_DIR / "yolov8m.pt"))
        )
        # Optimized ONNX model (small, imgsz=480) — preferred over .pt when present.
        # Generate with: python export_onnx.py
        self.yolo_small_onnx_path: Path = Path(
            os.getenv("YOLO_SMALL_ONNX_PATH", str(_PROJECT_DIR / "yolov8s.onnx"))
        )
        self.database_path: Path = Path(
            os.getenv("DATABASE_PATH", str(_PROJECT_DIR / "traffic_logs.db"))
        )
        self.video_source: str = os.getenv(
            "VIDEO_SOURCE_PATH", str(_PROJECT_DIR / "data" / "intersectie_romana.mp4")
        )
        self.q_table_path: Path = Path(
            os.getenv("Q_TABLE_PATH", str(_PROJECT_DIR / "q_table_medium_final.json"))
        )
        self.benchmark_results_path: Path = Path(
            os.getenv("BENCHMARK_RESULTS_PATH", str(_PROJECT_DIR / "benchmark_results.json"))
        )
        self.template_dir: Path = _PROJECT_DIR / "templates"
        self.uploads_dir: Path = _PROJECT_DIR / "uploads"

        # ── Flask ─────────────────────────────────────────────────────────
        self.flask_host: str = os.getenv("FLASK_HOST", "127.0.0.1")
        self.flask_port: int = int(os.getenv("FLASK_PORT", "5000"))
        self.flask_debug: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
        self.flask_threaded: bool = os.getenv("FLASK_THREADED", "true").lower() != "false"
        self.secret_key: str = os.getenv("SECRET_KEY", os.urandom(24).hex())

        # ── Video capture ─────────────────────────────────────────────────
        self.default_camera_index: int = int(os.getenv("DEFAULT_CAMERA_INDEX", "0"))
        self.target_fps: int = int(os.getenv("TARGET_FPS", "30"))
        self.video_encoding_quality: int = int(os.getenv("VIDEO_ENCODING_QUALITY", "60"))

        # ── YOLO detection ────────────────────────────────────────────────
        self.detection_confidence_threshold: float = float(
            os.getenv("DETECTION_CONFIDENCE_THRESHOLD", "0.20")
        )
        # 480 balances speed and accuracy for yolov8s on traffic scenes
        self.yolo_inference_size: int = int(os.getenv("YOLO_INFERENCE_SIZE", "480"))
        self.yolo_skip_interval: int = int(os.getenv("YOLO_SKIP_INTERVAL", "3"))

        # Per-class minimum confidence — keys must match values in class_mapping
        self.class_min_confidence: Dict[str, float] = {
            "person":    float(os.getenv("CONF_PERSON",    "0.65")),
            "motorbike": float(os.getenv("CONF_MOTORBIKE", "0.40")),
        }

        # Larger inference size for aerial/drone sources where vehicles are small.
        # Switch at runtime via change_video_source(inference_size=1280).
        self.yolo_aerial_inference_size: int = int(
            os.getenv("YOLO_AERIAL_INFERENCE_SIZE", "1280")
        )

        # Label consolidation (YOLO label → internal category)
        self.class_mapping: Dict[str, str] = {
            "car":        "car",
            "truck":      "car",
            "bus":        "car",
            "van":        "car",
            "motorcycle": "motorbike",
            "motorbike":  "motorbike",
            "bicycle":    "motorbike",
            "person":     "person",
        }

        # ── Traffic monitoring ────────────────────────────────────────────
        self.congestion_threshold_vehicles: int = int(
            os.getenv("CONGESTION_THRESHOLD", "25")
        )

        # ── Database ──────────────────────────────────────────────────────
        self.db_logging_interval: int = int(os.getenv("DB_LOGGING_INTERVAL", "10"))
        self.db_connection_timeout: int = int(os.getenv("DB_CONNECTION_TIMEOUT", "10"))
        self.db_max_retries: int = int(os.getenv("DB_MAX_RETRIES", "3"))
        self.db_retry_delay: int = int(os.getenv("DB_RETRY_DELAY", "5"))

        # ── Logging ───────────────────────────────────────────────────────
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        self.log_format: str = os.getenv(
            "LOG_FORMAT",
            "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Settings(host={self.flask_host}:{self.flask_port}, "
            f"model={self.yolo_model_path.name}, "
            f"db={self.database_path.name})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton — imported everywhere, loaded exactly once
# ---------------------------------------------------------------------------
settings: Settings = Settings()


# ---------------------------------------------------------------------------
# Path validation — called once at startup before any threads launch
# ---------------------------------------------------------------------------
def validate_paths() -> list[str]:
    """
    Check that all critical files/directories exist.

    Returns:
        List of error strings. Empty list means all paths are valid.
    """
    errors: list[str] = []

    # Accept ONNX small OR the configured .pt — either is enough to start
    onnx_ok = settings.yolo_small_onnx_path.exists()
    pt_ok   = settings.yolo_model_path.exists()
    if not onnx_ok and not pt_ok:
        errors.append(
            f"No YOLO model found. Checked:\n"
            f"  ONNX (fast): {settings.yolo_small_onnx_path}\n"
            f"  PT  (fallback): {settings.yolo_model_path}\n"
            f"  Run `python export_onnx.py` or set YOLO_MODEL_PATH / YOLO_SMALL_ONNX_PATH."
        )

    if not settings.database_path.parent.exists():
        errors.append(
            f"Database directory does not exist: {settings.database_path.parent}"
        )

    if not settings.template_dir.exists():
        errors.append(f"Templates directory not found: {settings.template_dir}")

    # Video source is not critical — capture thread falls back to camera
    video = settings.video_source
    if not video.startswith(("http", "https", "rtsp")) and not Path(video).exists():
        # Non-fatal: captured thread will use the default camera
        pass  # logged as warning by VideoCaptureWorker, not a startup blocker

    return errors
