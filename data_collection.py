# Data Collection Script - Integrated Camera Feed & Data Logging
import cv2
import time
import os
import threading
import logging
from datetime import datetime
from typing import Tuple, Optional
import numpy as np

# Import configuration
from config import (
    DEFAULT_CAMERA_INDEX, VIDEO_SOURCE_PATH, TARGET_FPS,
    DATA_COLLECTION_OUTPUT_DIR, DATA_COLLECTION_ENABLE_STORAGE,
    DATA_COLLECTION_FPS_LIMIT, LOG_LEVEL, LOG_FORMAT
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Integrated data collector that captures video frames and metadata
    Syncs with the main TrafficSystem for processing
    """

    def __init__(self, 
                 video_source: int = None,
                 frame_output_dir: Optional[str] = None,
                 fps_limit: int = None,
                 enable_local_storage: bool = None):
        """
        Initialize data collector
        
        Args:
            video_source: Camera index (0 for default) or path to video file (None uses config)
            frame_output_dir: Directory to save frames (None = uses config)
            fps_limit: Maximum FPS to capture at (None = uses config)
            enable_local_storage: Whether to save frames locally (None = uses config)
        """
        # Use config defaults if not provided
        self.video_source = video_source if video_source is not None else DEFAULT_CAMERA_INDEX
        self.frame_output_dir = frame_output_dir or DATA_COLLECTION_OUTPUT_DIR
        self.fps_limit = fps_limit or DATA_COLLECTION_FPS_LIMIT
        self.enable_local_storage = enable_local_storage if enable_local_storage is not None else DATA_COLLECTION_ENABLE_STORAGE
        self.is_running = False
        self.lock = threading.Lock()

        # Create output directory if needed
        if self.enable_local_storage and self.frame_output_dir:
            os.makedirs(self.frame_output_dir, exist_ok=True)
            logger.info(f"Frame storage enabled at: {self.frame_output_dir}")

        # Metrics
        self.frames_captured = 0
        self.frames_saved = 0
        self.last_frame_time = None
        self.camera_fps = TARGET_FPS

    def initialize_camera(self) -> Optional[cv2.VideoCapture]:
        """
        Initialize video capture source
        
        Returns:
            cv2.VideoCapture object or None if initialization fails
        """
        try:
            if isinstance(self.video_source, str):
                # It's a file path
                if not os.path.exists(self.video_source):
                    logger.error(f"Video file not found: {self.video_source}")
                    return None
                logger.info(f"Opening video file: {self.video_source}")
            else:
                # It's a camera index
                logger.info(f"Opening camera device: {self.video_source}")

            cap = cv2.VideoCapture(self.video_source)
            
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {self.video_source}")
                return None

            # Get camera properties
            self.camera_fps = cap.get(cv2.CAP_PROP_FPS)
            if self.camera_fps <= 0 or self.camera_fps > 120:
                self.camera_fps = 30
                cap.set(cv2.CAP_PROP_FPS, 30)

            logger.info(f"Camera initialized: FPS={self.camera_fps}")
            return cap

        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return None

    def capture_frame(self, cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray], dict]:
        """
        Capture single frame with metadata
        
        Args:
            cap: cv2.VideoCapture object
            
        Returns:
            Tuple of (success, frame, metadata)
        """
        try:
            success, frame = cap.read()
            
            if success:
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "frame_number": self.frames_captured,
                    "frame_shape": frame.shape,
                    "frame_size_mb": frame.nbytes / (1024 * 1024)
                }
                self.frames_captured += 1
                return success, frame, metadata

            logger.warning("Failed to read frame from source")
            return False, None, {}

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return False, None, {}

    def save_frame(self, frame: np.ndarray, frame_number: int) -> bool:
        """
        Save frame to local storage with timestamp
        
        Args:
            frame: Frame to save
            frame_number: Frame number/ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_local_storage or not self.frame_output_dir:
            return False

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{timestamp}_{frame_number:06d}.jpg"
            filepath = os.path.join(self.frame_output_dir, filename)

            success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            if success:
                self.frames_saved += 1
                if self.frames_saved % 100 == 0:
                    logger.info(f"Saved {self.frames_saved} frames to disk")
                return True
            else:
                logger.warning(f"Failed to save frame: {filepath}")
                return False

        except Exception as e:
            logger.error(f"Error saving frame: {e}")
            return False

    def collect_stream(self, callback=None) -> None:
        """
        Main data collection loop
        Captures frames continuously and optionally processes them via callback
        
        Args:
            callback: Function to call with (frame, metadata) - typically from TrafficSystem
        """
        cap = self.initialize_camera()
        if cap is None:
            logger.error("Cannot start data collection without valid video source")
            return

        self.is_running = True
        frame_delay = 1.0 / self.fps_limit
        logger.info(f"Data collection started - FPS limit: {self.fps_limit}")

        try:
            while self.is_running:
                start_time = time.time()

                success, frame, metadata = self.capture_frame(cap)

                if not success:
                    # For video files, loop back to start
                    if isinstance(self.video_source, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        logger.info("Video file looped")
                    continue

                # Save locally if enabled
                if self.enable_local_storage:
                    self.save_frame(frame, metadata["frame_number"])

                # Process via callback (typically sends to detection system)
                if callback and callable(callback):
                    try:
                        callback(frame, metadata)
                    except Exception as e:
                        logger.error(f"Error in callback: {e}")

                # Rate limiting
                elapsed = time.time() - start_time
                sleep_time = max(0.001, frame_delay - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Error in collection loop: {e}")
        finally:
            self.stop()
            cap.release()
            logger.info(f"Data collection stopped. Frames captured: {self.frames_captured}, saved: {self.frames_saved}")

    def get_stats(self) -> dict:
        """Get collection statistics"""
        with self.lock:
            return {
                "frames_captured": self.frames_captured,
                "frames_saved": self.frames_saved,
                "is_running": self.is_running,
                "camera_fps": self.camera_fps
            }

    def stop(self) -> None:
        """Stop data collection"""
        self.is_running = False
        logger.info("Data collection stop signal sent")


def collect_data(video_source: int = 0, output_dir: Optional[str] = None) -> None:
    """
    Standalone function to collect data from camera or video file
    
    Args:
        video_source: Camera index (0) or path to video file
        output_dir: Optional directory to save frames
    """
    collector = DataCollector(
        video_source=video_source,
        frame_output_dir=output_dir,
        enable_local_storage=output_dir is not None
    )

    try:
        collector.collect_stream()
    except KeyboardInterrupt:
        logger.info("Data collection interrupted by user")
        collector.stop()


if __name__ == '__main__':
    # Example usage - collect from camera and save frames
    output_directory = os.path.join(os.path.dirname(__file__), 'collected_frames')
    collect_data(video_source=0, output_dir=output_directory)