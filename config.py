# Configuration File - Smart Traffic Management System
# Centralized settings for all components

import os

# ============================================================
# APPLICATION PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))

# Model path
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, 'yolo26m.pt')

# Database path
DATABASE_PATH = os.path.join(BASE_DIR, 'traffic_logs.db')

# Video source (path to video file)
VIDEO_SOURCE_PATH = os.path.join(BASE_DIR, 'data', 'intersectie_romana.mp4')
#VIDEO_SOURCE_PATH = "58&mt=1773589281&fvip=2&fexp=51565116%2C51565682%2C51791334&c=ANDROID_VR&txp=5309224&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cxpc%2Cpcm2%2Cbui%2Cspc%2Cvprv%2Csvpuc%2Cmime%2Crqh%2Ccnr%2Cratebypass%2Cdur%2Clmt&sig=AHEqNM4wRQIgIyu3mkLi6WG1fVlnuGVqs7uWjTO-npDaB0454q8KnKoCIQDpURcLUF-vlTmaphLj2-HqMZP493Ub5rQ5jrmKo2Kd1Q%3D%3D&lsparams=cps%2Cmet%2Cmh%2Cmm%2Cmn%2Cms%2Cmv%2Cmvi%2Cpl%2Crms%2Cinitcwndbps&lsig=APaTxxMwRQIgb-bnPeJboee753YmcgKdaT-hmbvaXog0Mv2dxXm7iRwCIQCV4Jr6Bsp2gCRNSk9Ixy-9Z0YpgfbuTcYSF1Jh64nyTA%3D%3D"
#YOUTUBE_LIVE_URL = "https://www.youtube.com/watch?v=3wI4xzieiy4"
#USE_YOUTUBE_LIVE = True
#VIDEO_SOURCE_PATH = "https://www.youtube.com/watch?v=CCp4KhzcQXE"
# Data collection output directory (optional)
DATA_COLLECTION_OUTPUT_DIR = os.path.join(BASE_DIR, 'collected_frames') 

# ============================================================
# FLASK APPLICATION SETTINGS
# ============================================================
FLASK_HOST = os.getenv('FLASK_HOST', '127.0.0.1')
FLASK_PORT = 5000
FLASK_DEBUG = False
FLASK_THREADED = True

# ============================================================
# VIDEO & CAPTURE SETTINGS
# ============================================================
# Default camera device index (0 = primary camera)
DEFAULT_CAMERA_INDEX = 0

# Target FPS for video capture
TARGET_FPS = 30

# Frame rate cap for streaming
STREAM_MAX_FPS = 50

# Video frame encoding quality (1-100)
VIDEO_ENCODING_QUALITY = 80

# ============================================================
# OBJECT DETECTION (YOLO) SETTINGS
# ============================================================
# Confidence threshold for detections (0.0-1.0)
DETECTION_CONFIDENCE_THRESHOLD = 0.20

# Inference resolution (must be multiple of 32: 320, 416, 608, 640, etc.)
YOLO_INFERENCE_SIZE = 640

# Skip frames for optimization
# YOLO runs every Nth frame (e.g., 3 = run every 3rd frame)
YOLO_SKIP_INTERVAL = 3

# Per-class minimum confidence thresholds
# These are applied after general threshold to filter false positives
CLASS_MIN_CONFIDENCE = {
    "person":     float(os.getenv("CONF_PERSON",     "0.65")),  # Higher: avoids pole/post misdetection
    "motorcycle": float(os.getenv("CONF_MOTORCYCLE", "0.40")),  # Lower: motorbike detections are noisy
    "bicycle":    float(os.getenv("CONF_BICYCLE",    "0.45")),
}

# Class mapping (consolidate multiple labels to fewer categories)
CLASS_MAPPING = {
    "car": "car",
    "truck": "car",
    "bus": "car",
    "van": "car",
    "motorcycle": "motorbike",
    "motorbike": "motorbike",
    "bicycle": "motorbike",
    "person": "person",
}

# Tracked object classes
TRACKED_CLASSES = ["car", "motorbike", "person"]

# ============================================================
# TRAFFIC MONITORING SETTINGS
# ============================================================
# Maximum vehicles for a highway segment before congestion alert
CONGESTION_THRESHOLD_VEHICLES = 25

# Congestion level thresholds (percentage)
CONGESTION_LEVEL = {
    "free_flow": 30,      # 0-30%: Free flow
    "moderate": 70,       # 30-70%: Moderate congestion
    "heavy": 100,         # 70-100%: Heavy congestion
}

# Moving average window size (number of frames to average)
MOVING_AVERAGE_WINDOW = 5

# ============================================================
# DATABASE SETTINGS
# ============================================================
# Database logging interval (seconds)
DB_LOGGING_INTERVAL = 10

# Database connection timeout (seconds)
DB_CONNECTION_TIMEOUT = 10

# Database retry policy
DB_MAX_RETRIES = 3
DB_RETRY_DELAY = 5

# Enable WAL (Write-Ahead Logging) mode for better concurrency
DB_ENABLE_WAL = True



# ============================================================
# DATA COLLECTION SETTINGS
# ============================================================
# Enable local frame storage
DATA_COLLECTION_ENABLE_STORAGE = False

# FPS limit for data collection
DATA_COLLECTION_FPS_LIMIT = 30

# ============================================================
# LOGGING SETTINGS
# ============================================================
# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = 'INFO'

# Log format
LOG_FORMAT = '[%(asctime)s] %(levelname)s: %(message)s'

# Log file path (None = console only)
LOG_FILE = None  # Set to a path to enable file logging

# ============================================================
# THREADING SETTINGS
# ============================================================
# Thread daemon mode (True = threads exit when main thread exits)
THREAD_DAEMON = True

# Capture thread error tolerance (max consecutive errors before reconnect)
CAPTURE_ERROR_TOLERANCE = 20

# ============================================================
# PERFORMANCE TUNING
# ============================================================
# Enable performance monitoring
ENABLE_PERFORMANCE_MONITORING = False

# Number of detections to keep in rolling history
DETECTION_HISTORY_SIZE = 100

# ============================================================
# ERROR MESSAGES
# ============================================================
ERROR_MESSAGES = {
    "yolo_import_error": "ImportError: ultralytics library not found. Install with: pip install ultralytics",
    "yolo_file_not_found": "CRITICAL ERROR: YOLO model not found at {path}",
    "yolo_load_error": "CRITICAL ERROR loading YOLO model: {error}",
    "db_init_error": "Database initialization error: {error}",
    "video_source_error": "Failed to open video source: {source}",
    "camera_init_error": "Failed to initialize camera device",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_paths():
    """Validate that all required paths are accessible"""
    from pathlib import Path
    
    errors = []
    
    # Check YOLO model
    if not Path(YOLO_MODEL_PATH).exists():
        errors.append(f"YOLO model not found at {YOLO_MODEL_PATH}")
    
    # Check video source (not critical - can use camera instead)
    if not Path(VIDEO_SOURCE_PATH).exists():
        print(f"[WARNING] Video file not found at {VIDEO_SOURCE_PATH}, will use camera instead")
    
    return errors


def get_config_summary():
    """Return a summary of current configuration"""
    return {
        "app_name": "Smart Traffic Management System",
        "flask_port": FLASK_PORT,
        "yolo_model": YOLO_MODEL_PATH,
        "database": DATABASE_PATH,
        "detection_threshold": DETECTION_CONFIDENCE_THRESHOLD,
        "skip_frames": YOLO_SKIP_INTERVAL,
        "db_interval_seconds": DB_LOGGING_INTERVAL,
        "congestion_threshold": CONGESTION_THRESHOLD_VEHICLES,
    }


if __name__ == '__main__':
    # Print configuration on direct execution
    import json
    print("=" * 60)
    print("Smart Traffic Management System - Configuration")
    print("=" * 60)
    print(json.dumps(get_config_summary(), indent=2))
    print("=" * 60)
    
    # Validate paths
    errors = validate_paths()
    if errors:
        print("\n[ERRORS]")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n[OK] All paths validated successfully")
