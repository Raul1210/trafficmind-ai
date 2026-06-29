"""
TrafficSystem — the central orchestrator.

Owns and lifecycle-manages:
  - VideoCaptureWorker   (video ingestion + YOLO inference)
  - DatabaseLogger       (SQLite write loop)
  - TrafficSignalAgent   (Q-learning RL state)
  - rl_lock              (protects concurrent RL state access from routes)

This module is intentionally free of Flask concerns — it can be instantiated
and tested independently of the HTTP layer.
"""
from __future__ import annotations

import threading
import time

from typing import Optional

from traffic_monitor.agent.q_learning import (
    PHASE_NAMES,
    TrafficSignalAgent,
)
from traffic_monitor.alerts.email_alerts import EmailAlertManager
from traffic_monitor.config import Settings
from traffic_monitor.core.capture import VideoCaptureWorker
from traffic_monitor.core.database import DatabaseLogger, init_db
from traffic_monitor.logger import get_logger

logger = get_logger(__name__)


class TrafficSystem:
    """
    Creates, starts, and cleanly shuts down all background subsystems.

    Usage inside create_app():
        system = TrafficSystem(settings)
        atexit.register(system.shutdown)
    """

    def __init__(self, cfg: Settings) -> None:
        self._cfg = cfg
        self.start_time: float = time.time()

        # ── Serialises concurrent source-switch requests ───────────────────
        self._source_lock = threading.Lock()

        # ── Q-learning agent + shared state dict ──────────────────────────
        self.rl_lock = threading.Lock()
        self.rl_agent = TrafficSignalAgent()
        self._load_agent()

        self.rl_state: dict = {
            "phase":          0,
            "phase_name":     PHASE_NAMES[0],
            "phase_duration": 0,
            "q_values":       [0.0, 0.0, 0.0, 0.0],
            "confidence":     0.0,
            "queues":         {"north": 0, "south": 0, "east": 0, "west": 0},
            "reward":         0.0,
            "last_updated":   0,
            # ── Scene classification ──────────────────────────────────────
            # Updated asynchronously by _classify_scene_background() each time
            # a new video source is loaded.  rl_active drives the UI toggle.
            "scene_type":            "uncertain",
            "scene_confidence":      0.0,
            "scene_dominant_angles": [],
            "rl_active":             True,   # False only when scene_type=="single_flow"
        }

        # ── Database ──────────────────────────────────────────────────────
        if not init_db(str(cfg.database_path), cfg.db_connection_timeout):
            logger.warning("Database init failed — data logging may be unavailable")

        # ── Email alert manager ───────────────────────────────────────────
        self.email_alerts = EmailAlertManager(str(cfg.database_path))

        # ── Background threads ────────────────────────────────────────────
        self.capture   = VideoCaptureWorker(cfg)
        self.db_logger = DatabaseLogger(cfg, self._get_db_snapshot, self.email_alerts)

        self.capture.start()
        self.db_logger.start()
        logger.info("TrafficSystem started — all subsystems running")

        # Kick off initial scene classification for the startup source
        self._start_scene_classifier(self.capture)

    # ── Agent helpers ─────────────────────────────────────────────────────

    def _load_agent(self) -> None:
        path = str(self._cfg.q_table_path)
        try:
            self.rl_agent.load(path)
            self.rl_agent.epsilon = 0.05
            logger.info(f"Q-agent loaded from {path}")
        except FileNotFoundError:
            logger.info("No saved Q-table found — agent starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load Q-agent: {e} — starting fresh")

    # ── Snapshot for DatabaseLogger ────────────────────────────────────────

    def _get_db_snapshot(self) -> tuple[int, float, float]:
        """
        Called by DatabaseLogger (its own thread) once per logging interval.
        Returns (vehicles_count, avg_confidence, congestion_percentage).
        get_snapshot() acquires the capture lock atomically.
        """
        smoothed, avg_conf, congestion = self.capture.get_snapshot()
        vehicles = smoothed.get("car", 0) + smoothed.get("motorbike", 0)
        return vehicles, avg_conf, congestion

    # ── Forwarded status properties (read by routes) ───────────────────────

    @property
    def db_connected(self) -> bool:
        return self.db_logger.db_connected

    @property
    def db_row_count(self) -> int:
        return self.db_logger.db_row_count

    # ── Scene classification ───────────────────────────────────────────────

    def _start_scene_classifier(self, worker) -> None:
        """Spawn a daemon thread that classifies the scene for *worker*."""
        threading.Thread(
            target=self._classify_scene_background,
            args=(worker,),
            daemon=True,
            name="SceneClassifier",
        ).start()

    def _classify_scene_background(self, worker) -> None:
        """
        Background daemon: analyse vehicle motion from *worker* for up to 10 s,
        then write scene_type / rl_active into rl_state (under rl_lock).

        Holding a reference to *worker* (not self.capture) ensures that a second
        source switch cancels classification for the old source gracefully: the
        old worker stops producing new boxes, the classifier times out, writes
        "uncertain", and exits.  The new switch already started a fresh thread.
        """
        from traffic_monitor.detection.scene_classifier import SceneClassifier

        clf = SceneClassifier(window_frames=25, max_match_dist_px=90)

        # Brief stabilisation delay before the new stream produces frames
        time.sleep(1.0)

        deadline = time.time() + 10.0
        last_box_id: int = id(None)

        while time.time() < deadline:
            try:
                with worker.lock:
                    boxes    = worker.boxes
                    box_id   = id(boxes)
            except Exception:
                break

            if box_id != last_box_id:
                clf.update(list(boxes))
                last_box_id = box_id

            time.sleep(0.08)   # ~12.5 Hz — fast enough to catch skip_frames=2

            if clf.is_ready():
                time.sleep(1.5)  # collect a bit more data for stability
                break

        result = clf.classify()
        logger.info(
            f"SceneClassifier result: {result['scene_type']} "
            f"(conf={result['confidence']:.2f}, "
            f"vectors={result['vector_count']}, "
            f"angles={result['dominant_angles']})"
        )
        with self.rl_lock:
            self.rl_state["scene_type"]            = result["scene_type"]
            self.rl_state["scene_confidence"]      = result["confidence"]
            self.rl_state["scene_dominant_angles"] = result["dominant_angles"]
            # Disable RL only for confirmed single-flow roads
            self.rl_state["rl_active"] = result["scene_type"] != "single_flow"

    # ── Video source switching ─────────────────────────────────────────────

    def change_video_source(
        self,
        new_source: str,
        inference_size: Optional[int] = None,
    ) -> None:
        """
        Thread-safe video source switch.

        Steps:
          1. Acquire _source_lock to serialise concurrent switch requests.
          2. Stop the running VideoCaptureWorker (signals + joins with timeout).
          3. Reset FluidDetector state so ghost boxes from the old source don't
             bleed into the first frames of the new source, and optionally
             update inference_size (pass a larger value for aerial sources).
          4. Update cfg.video_source so the new worker picks up the value
             via _resolve_source().
          5. Create and start a fresh VideoCaptureWorker.

        The DatabaseLogger is unaffected — its get_snapshot callback delegates
        to self.capture, which will point to the new worker after the swap.
        The /video_feed generator will return None frames for the brief
        transition gap, which is handled by its existing error-counting logic.
        """
        from traffic_monitor.detection.yolo_detector import fluid_detector

        with self._source_lock:
            logger.info(f"Video source switch requested: {new_source!r}"
                        + (f" (inference_size={inference_size})" if inference_size else ""))

            old = self.capture
            old.stop()
            old.join(timeout=8)
            if old.is_alive():
                logger.warning("Old capture worker did not stop cleanly — proceeding anyway")

            # Clear ghost boxes / cached frames from the previous source.
            # Also updates inference_size when explicitly provided.
            fluid_detector.reset_state(new_inference_size=inference_size)

            self._cfg.video_source = new_source

            self.capture = VideoCaptureWorker(self._cfg)
            self.capture.start()
            logger.info(f"New capture worker started for source: {new_source!r}")

            # Reset scene type to "uncertain" immediately; the background thread
            # below will overwrite it with the real classification once ready.
            with self.rl_lock:
                self.rl_state["scene_type"]            = "uncertain"
                self.rl_state["scene_confidence"]      = 0.0
                self.rl_state["scene_dominant_angles"] = []
                self.rl_state["rl_active"]             = True

            self._start_scene_classifier(self.capture)

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """
        Gracefully stop all background threads.
        Registered with atexit in create_app() — called automatically on exit.
        """
        logger.info("Shutting down TrafficSystem...")
        self.capture.stop()
        self.capture.join(timeout=5)
        self.db_logger.stop()
        self.db_logger.join(timeout=5)
        logger.info("TrafficSystem shutdown complete")
