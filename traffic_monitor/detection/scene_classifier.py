"""
SceneClassifier — centroid-tracking motion analysis to distinguish
multi-branch intersections from single-flow roads (highway / arterial).

Input:  sequences of bounding-box lists already produced by FluidDetector
        or YOLODetector — each element: (x1, y1, x2, y2, label[, conf, …])
Output: {"scene_type", "confidence", "dominant_angles", "vector_count"}

Algorithm:
  1. For each consecutive frame pair, match boxes by nearest same-class centroid
     (max_match_dist_px threshold).
  2. Compute (dx, dy) displacement vectors; discard static vehicles (< MIN_MOTION_PX).
  3. Convert to angles with atan2; accumulate over a rolling window.
  4. Bin angles into 8 sectors of 45°; classify with two-stage check:
       single_flow  — primary:  ≥ 80 % in best 135° arc (3 adjacent bins)
                      fallback: ≥ 65 % in best  90° arc (2 adjacent bins)
       intersection — everything else with sufficient data
       uncertain    — fewer than MIN_VECTORS
  Note: the 135° primary check catches dual-carriageway highways viewed from
  above where both directions are visible (~135° spread), which the 90° check
  alone mis-classifies as intersection.
"""
from __future__ import annotations

import math

from traffic_monitor.logger import get_logger

logger = get_logger(__name__)


class SceneClassifier:
    """
    Classify the current video source as "intersection", "single_flow",
    or "uncertain" based on the aggregate direction of vehicle motion.

    Lifecycle:
        sc = SceneClassifier()
        sc.reset()                       # call on every source switch
        for boxes in frame_box_stream:
            sc.update(boxes)
        result = sc.classify()           # → dict
    """

    # ---------- tuneable constants ----------
    _SINGLE_FLOW_ARC135_THRESH: float = 0.80  # fraction of vectors in best 135° arc (primary)
    _SINGLE_FLOW_ARC_THRESH:    float = 0.65  # fraction of vectors in best  90° arc (fallback)
    _MIN_VECTORS:               int   = 15    # minimum vectors before any verdict
    _MIN_MOTION_PX:             float = 3.0   # ignore displacements smaller than this

    def __init__(
        self,
        window_frames: int = 25,
        max_match_dist_px: int = 90,
    ) -> None:
        """
        Args:
            window_frames:      how many frames' worth of vectors to keep
            max_match_dist_px:  centroid distance ceiling for box matching
        """
        self._window   = window_frames
        self._max_dist = max_match_dist_px
        # (cx, cy, label) tuples from the previous frame
        self._prev: list[tuple[float, float, str]] = []
        # rolling history of motion-vector angles (degrees, 0–360)
        self._angles: list[float] = []

    # ── Public API ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear accumulated state. Call immediately after a source switch."""
        self._prev   = []
        self._angles = []

    def update(self, boxes: list[tuple]) -> None:
        """
        Process one frame's bounding-box list.
        Each box must be (x1, y1, x2, y2, label[, conf, …]).
        Safe to call with an empty list (no-op).
        """
        curr: list[tuple[float, float, str]] = []
        for box in boxes:
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            label = box[4] if len(box) > 4 else "car"
            curr.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, label))

        if self._prev and curr:
            matched: set[int] = set()
            for pcx, pcy, plabel in self._prev:
                best_dist, best_j = float("inf"), -1
                for j, (ccx, ccy, clabel) in enumerate(curr):
                    if j in matched or clabel != plabel:
                        continue
                    d = math.hypot(ccx - pcx, ccy - pcy)
                    if d < best_dist:
                        best_dist, best_j = d, j

                if best_j >= 0 and best_dist <= self._max_dist:
                    dx = curr[best_j][0] - pcx
                    dy = curr[best_j][1] - pcy
                    mag = math.hypot(dx, dy)
                    if mag >= self._MIN_MOTION_PX:
                        # Standard math angle: right=0°, up=90° (dy flipped for image coords)
                        angle = math.degrees(math.atan2(-dy, dx)) % 360
                        self._angles.append(angle)
                    matched.add(best_j)

        # Rolling window: keep at most window * (avg ~8 vehicles/frame)
        cap = self._window * 10
        if len(self._angles) > cap:
            self._angles = self._angles[-cap:]

        self._prev = curr

    def is_ready(self) -> bool:
        """True when enough vectors have been collected for a reliable verdict."""
        return len(self._angles) >= self._MIN_VECTORS

    def classify(self) -> dict:
        """
        Returns:
            scene_type:      "intersection" | "single_flow" | "uncertain"
            confidence:      float 0–1
            dominant_angles: list[float] — bin-centre angles of top cluster(s), degrees
            vector_count:    int — total motion vectors accumulated so far
        """
        n = len(self._angles)

        if n < self._MIN_VECTORS:
            return {
                "scene_type":      "uncertain",
                "confidence":      0.0,
                "dominant_angles": [],
                "vector_count":    n,
            }

        # ── Bin into 8 sectors of 45° ─────────────────────────────────────
        bins = [0] * 8
        for a in self._angles:
            bins[int(a / 45) % 8] += 1

        # ── Compute both arc metrics from the same bins ────────────────────
        best_arc135 = max(bins[i] + bins[(i + 1) % 8] + bins[(i + 2) % 8] for i in range(8))
        arc_frac135 = best_arc135 / n

        best_arc = max(bins[i] + bins[(i + 1) % 8] for i in range(8))
        arc_frac = best_arc / n

        # ── Single-flow check: 135° arc (primary) ─────────────────────────
        # A straight road with both directions visible (dual carriageway seen
        # from a bridge) concentrates traffic in a ~135° cone around the road
        # axis — the 90° window alone is too narrow to catch this case.
        if arc_frac135 >= self._SINGLE_FLOW_ARC135_THRESH:
            best135_i = max(range(8), key=lambda i: bins[i] + bins[(i + 1) % 8] + bins[(i + 2) % 8])
            top_bin = max(
                (best135_i, (best135_i + 1) % 8, (best135_i + 2) % 8),
                key=lambda i: bins[i],
            )
            conf = min(1.0, (arc_frac135 - self._SINGLE_FLOW_ARC135_THRESH)
                       / (1.0 - self._SINGLE_FLOW_ARC135_THRESH + 1e-9))
            logger.debug(
                f"SceneClassifier single_flow (135-arc): arc_frac135={arc_frac135:.2f} "
                f"conf={conf:.2f} top_bin={top_bin}"
            )
            return {
                "scene_type":      "single_flow",
                "confidence":      round(conf, 3),
                "dominant_angles": [round(top_bin * 45 + 22.5, 1)],
                "vector_count":    n,
            }

        # ── Single-flow check: 90° arc (fallback for strict unidirectional) ─
        top_bin = max(range(8), key=lambda i: bins[i])
        if arc_frac >= self._SINGLE_FLOW_ARC_THRESH:
            excess = arc_frac - self._SINGLE_FLOW_ARC_THRESH
            span   = 1.0 - self._SINGLE_FLOW_ARC_THRESH + 1e-9
            conf   = min(1.0, excess / span)
            logger.debug(
                f"SceneClassifier single_flow (90-arc): arc_frac={arc_frac:.2f} "
                f"conf={conf:.2f} top_bin={top_bin}"
            )
            return {
                "scene_type":      "single_flow",
                "confidence":      round(conf, 3),
                "dominant_angles": [round(top_bin * 45 + 22.5, 1)],
                "vector_count":    n,
            }

        # ── Intersection (everything that is NOT single_flow) ─────────────
        # Covers: 4-way intersections, roundabouts, two-way boulevards — any
        # scene where vehicles move in multiple distinct directions.
        # Confidence scales with how far the distribution is from single_flow:
        # a perfectly uniform spread (arc_frac ≈ 0.25) gives conf close to 1.
        conf = min(1.0, (self._SINGLE_FLOW_ARC_THRESH - arc_frac)
                   / (self._SINGLE_FLOW_ARC_THRESH - 0.25 + 1e-9))
        conf = max(0.0, conf)

        ranked = sorted(range(8), key=lambda i: bins[i], reverse=True)
        top1, top2 = ranked[0], ranked[1]
        ang1 = top1 * 45 + 22.5
        ang2 = top2 * 45 + 22.5

        logger.debug(
            f"SceneClassifier intersection: arc_frac={arc_frac:.2f} conf={conf:.2f}"
        )
        return {
            "scene_type":      "intersection",
            "confidence":      round(conf, 3),
            "dominant_angles": [round(ang1, 1), round(ang2, 1)],
            "vector_count":    n,
        }
