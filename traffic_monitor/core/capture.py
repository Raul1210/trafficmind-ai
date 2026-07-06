"""
Video capture + real-time AI inference — producer/consumer architecture.

Threading model (TASK 6):
  _capture_loop  [sub-thread "FrameCapture"]
      Reads raw frames from the video source at full source FPS.
      Puts frames into _infer_queue (maxsize=2, non-blocking → drops stale frames).
      Does NOT run YOLO — never blocks on inference.

  _inference_loop  [main VideoCaptureWorker thread]
      Consumes frames from _infer_queue.
      Runs FluidDetector.process() — YOLO every (skip_frames+1) frames,
      cached annotated frame on skipped frames.
      Writes self._frame for the MJPEG stream.

Result: self._frame is updated at the rate inference+skipping allows.
With yolov8n.onnx (imgsz=416) this reaches 20–30 FPS on CPU.
The _infer_queue (maxsize=2) ensures the inference thread always
processes near-latest frames even if YOLO is momentarily slower.

Public API is identical to the previous single-threaded version.
"""
from __future__ import annotations

import collections
import os
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np

from traffic_monitor.config import Settings
from traffic_monitor.detection.yolo_detector import fluid_detector
from traffic_monitor.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# YouTube URL extraction helper (unchanged)
# ---------------------------------------------------------------------------

def _get_youtube_stream_url(url: str, socket_timeout: int = 10) -> Optional[str]:
    """Extract a direct stream URL from YouTube via yt-dlp."""
    try:
        import yt_dlp
        ydl_opts = {
            "format": "best[height<=720]/best",
            "quiet": True,
            "socket_timeout": socket_timeout,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_url: str = info["url"]
            logger.info("YouTube stream URL obtained successfully")
            return stream_url
    except Exception as e:
        logger.error(f"Error extracting YouTube URL: {e}")
        return None


# ---------------------------------------------------------------------------
# Main worker thread
# ---------------------------------------------------------------------------

class VideoCaptureWorker(threading.Thread):
    """
    Background thread: frame reading + YOLO detection + annotation.

    Public read-only state (access only via the methods below):
        counts          — current vehicle counts per class
        boxes           — current bounding boxes with labels + confidence
        avg_confidence  — mean detection confidence this inference cycle
        congestion_level — 0–100 %

    Public methods (all thread-safe):
        get_frame_bytes()        — JPEG bytes for the MJPEG stream
        get_snapshot()           — atomic (smoothed_counts, avg_conf, congestion)
        get_smoothed_counts()    — 5-frame moving-average counts
        get_directional_counts() — N/S/E/W split based on frame geometry
        get_congestion()         — current congestion percentage
        stop()                   — signal the thread to exit
    """

    def __init__(self, cfg: Settings) -> None:
        super().__init__(daemon=True, name="VideoCaptureWorker")
        self._cfg = cfg
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Raw frame queue: _capture_loop → _inference_loop
        # maxsize=2: small buffer keeps latency low; excess frames are dropped.
        self._infer_queue: queue.Queue = queue.Queue(maxsize=2)

        # Protected mutable state
        self._frame: Optional[np.ndarray] = None
        self._count_history: collections.deque = collections.deque(maxlen=5)

        # Publicly read by routes (always via lock or atomic methods below)
        self.counts: dict[str, int] = {"car": 0, "motorbike": 0, "person": 0}
        self.boxes: list[tuple] = []
        self.avg_confidence: float = 0.0
        self.congestion_level: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def lock(self) -> threading.Lock:
        """Exposed for compound atomic operations in system.py."""
        return self._lock

    def get_frame_bytes(self) -> Optional[bytes]:
        """Return the latest annotated frame as JPEG bytes, or None."""
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame.copy()
        ok, buffer = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, self._cfg.video_encoding_quality],
        )
        return buffer.tobytes() if ok else None

    def get_snapshot(self) -> tuple[dict[str, int], float, float]:
        """
        Atomically return (smoothed_counts, avg_confidence, congestion_level).
        Used by DatabaseLogger to avoid multiple lock acquisitions.
        """
        with self._lock:
            return self._smoothed_counts(), self.avg_confidence, self.congestion_level

    def get_smoothed_counts(self) -> dict[str, int]:
        with self._lock:
            return self._smoothed_counts()

    def get_directional_counts(self) -> dict[str, int]:
        """Split detected vehicles into N/S/E/W quadrants based on frame centre."""
        with self._lock:
            boxes = list(self.boxes)
            frame = self._frame

        if frame is None or not boxes:
            return {"north": 0, "south": 0, "east": 0, "west": 0}

        h, w = frame.shape[:2]
        mid_y, mid_x = h // 2, w // 2
        counts = {"north": 0, "south": 0, "east": 0, "west": 0}

        for box in boxes:
            if len(box) < 5 or box[4] == "person":
                continue
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            counts["north" if cy < mid_y else "south"] += 1
            counts["west"  if cx < mid_x else "east"]  += 1

        return counts

    def get_congestion(self) -> float:
        with self._lock:
            return self.congestion_level

    def stop(self) -> None:
        """Signal both threads to exit on the next iteration."""
        self._stop_event.set()

    # ── Thread orchestration (TASK 6) ────────────────────────────────────

    def run(self) -> None:
        """
        Start the fast capture sub-thread, then run the inference loop
        in this thread.
        """
        cap_thr = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="FrameCapture",
        )
        cap_thr.start()
        self._inference_loop()

    # ── Capture sub-thread ────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """
        Fast frame reader — reads the video source at full source FPS and
        feeds raw frames into _infer_queue.  Never touches YOLO.
        """
        try:
            source = self._resolve_source()
            cap, frame_delay = self._open_capture(source)
        except Exception as e:
            logger.error(f"Capture sub-thread failed to start: {e}")
            return

        frame_errors = 0
        max_errors   = 20

        while not self._stop_event.is_set():
            try:
                ok, raw = cap.read()

                if not ok or raw is None:
                    frame_errors += 1
                    if frame_errors > max_errors:
                        logger.warning("Too many frame read errors — reconnecting...")
                        frame_errors = 0
                        cap.release()
                        source = self._resolve_source()
                        try:
                            cap, frame_delay = self._open_capture(source)
                        except Exception as e:
                            logger.error(f"Reconnect failed: {e} — retrying in 10 s")
                            time.sleep(10)
                    else:
                        time.sleep(0.1)
                    continue

                # Format conversion (YouTube may deliver BGRA or grayscale)
                try:
                    if len(raw.shape) == 2:
                        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
                    elif raw.shape[2] == 4:
                        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                except cv2.error as e:
                    logger.warning(f"Frame conversion error: {e}, skipping frame")
                    continue

                frame_errors = 0

                # Non-blocking put — drop frame if inference is still busy.
                # This keeps latency low: the inference thread always gets
                # a near-current frame rather than a stale queue backlog.
                if not self._infer_queue.full():
                    self._infer_queue.put_nowait(raw)

                time.sleep(max(0.001, frame_delay - 0.005))

            except Exception as e:
                logger.error(f"Capture loop error: {e}")

        try:
            cap.release()
            logger.info("Video capture released")
        except Exception as e:
            logger.warning(f"Failed to release video capture: {e}")

    # ── Inference loop ────────────────────────────────────────────────────

    def _inference_loop(self) -> None:
        """
        YOLO inference — consumes raw frames from _infer_queue,
        runs FluidDetector (skip logic inside), updates self._frame.
        """
        while not self._stop_event.is_set():
            # Block until a frame is available (timeout avoids hanging on stop)
            try:
                raw = self._infer_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # ── FluidDetector: YOLO or cached result ─────────────────────
            try:
                annotated, new_counts, new_boxes, avg_conf = fluid_detector.process(raw)
            except Exception as e:
                logger.error(f"Detection error: {e}")
                continue

            # ── Update shared detection state ─────────────────────────────
            with self._lock:
                self.counts = {
                    k: new_counts.get(k, 0)
                    for k in ("car", "motorbike", "person")
                }
                self.boxes          = new_boxes
                self.avg_confidence = avg_conf
                self._count_history.append(dict(self.counts))
                vehicles = self.counts["car"] + self.counts["motorbike"]
                self.congestion_level = min(
                    (vehicles / self._cfg.congestion_threshold_vehicles) * 100,
                    100.0,
                )

            if annotated is not None:
                with self._lock:
                    self._frame = annotated

    # ── Private helpers ───────────────────────────────────────────────────

    def _smoothed_counts(self) -> dict[str, int]:
        """Compute 5-frame moving average. Caller must hold self._lock."""
        if not self._count_history:
            return {"car": 0, "motorbike": 0, "person": 0}
        n = len(self._count_history)
        return {
            key: round(sum(s.get(key, 0) for s in self._count_history) / n)
            for key in ("car", "motorbike", "person")
        }

    def _resolve_source(self) -> str | int:
        """Return the correct video source (URL, file path, or camera index)."""
        src = self._cfg.video_source
        if isinstance(src, str) and src.startswith(("http", "https", "rtsp")):
            if "youtube.com" in src or "youtu.be" in src:
                logger.info("YouTube source detected, extracting direct URL...")
                url = _get_youtube_stream_url(src)
                if url:
                    return url
                fallback = str(self._cfg.project_dir / "data" / "intersectie_romana.mp4")
                logger.error(f"Could not get YouTube URL, falling back to {fallback}")
                return fallback
            return src
        if os.path.exists(str(src)):
            return str(src)
        logger.warning(
            f"Video file not found at {src}, "
            f"using camera index {self._cfg.default_camera_index}"
        )
        return self._cfg.default_camera_index

    def _open_capture(self, source) -> tuple[cv2.VideoCapture, float]:
        """Open cv2.VideoCapture and return (cap, frame_delay_seconds)."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not (0 < fps <= 120):
            fps = self._cfg.target_fps
        logger.info(f"Video capture opened — source={source!r}  FPS={fps:.1f}")
        return cap, 1.0 / fps
