"""
test_scene_classifier.py — Manual test for SceneClassifier.

Run from the project root:
    python test_scene_classifier.py

Tests:
    A — Real video (traffic.mp4 / Piata Romana)  -> expected: intersection
    B — Synthetic single-flow (highway sim)      -> expected: single_flow
    C — Synthetic intersection (N-S + E-W)       -> expected: intersection
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import cv2

from traffic_monitor.detection.scene_classifier import SceneClassifier
from traffic_monitor.detection.yolo_detector import fluid_detector
from traffic_monitor.config import settings


# ─────────────────────────────────────────────────────────────
#  Test A: real video with FluidDetector inference
# ─────────────────────────────────────────────────────────────

def test_real_video(video_path: str, n_frames: int = 80) -> dict:
    print(f"\n{'='*60}")
    print(f"Test A: Real video — {os.path.basename(video_path)}")
    print(f"{'='*60}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return {}

    clf = SceneClassifier(window_frames=25, max_match_dist_px=90)
    fluid_detector.reset_state()

    frame_idx = 0
    while frame_idx < n_frames:
        ok, frame = cap.read()
        if not ok:
            break

        _annotated, counts, boxes, _conf = fluid_detector.process(frame)
        clf.update(boxes)

        if frame_idx % 15 == 0 and frame_idx > 0:
            r = clf.classify()
            total_v = counts.get("car", 0) + counts.get("motorbike", 0)
            print(
                f"  frame {frame_idx:3d} | boxes={len(boxes):2d} | "
                f"total_v={total_v} | vectors={r['vector_count']:3d} | "
                f"→ {r['scene_type']} (conf={r['confidence']:.2f})"
            )

        frame_idx += 1

    cap.release()

    final = clf.classify()
    print(f"\n  FINAL ({frame_idx} frames processed):")
    print(f"    scene_type:      {final['scene_type']}")
    print(f"    confidence:      {final['confidence']:.3f}")
    print(f"    dominant_angles: {final['dominant_angles']} deg")
    print(f"    vector_count:    {final['vector_count']}")
    return final


# ─────────────────────────────────────────────────────────────
#  Test B: synthetic single-flow (all vehicles move +x)
# ─────────────────────────────────────────────────────────────

def test_synthetic_single_flow(n_frames: int = 40) -> dict:
    print(f"\n{'='*60}")
    print("Test B: Synthetic single-flow (highway — all vehicles move right)")
    print(f"{'='*60}")

    clf = SceneClassifier(window_frames=20, max_match_dist_px=90)

    for frame_i in range(n_frames):
        boxes = []
        for j in range(6):
            x = (50 + frame_i * 20 + j * 110) % 1800
            y = 80 + j * 70
            boxes.append((x, y, x + 70, y + 45, "car", 0.85))
        clf.update(boxes)

        if frame_i % 8 == 7:
            r = clf.classify()
            print(
                f"  frame {frame_i+1:2d} | vectors={r['vector_count']:3d} | "
                f"→ {r['scene_type']} (conf={r['confidence']:.2f})"
            )

    final = clf.classify()
    print(f"\n  FINAL ({n_frames} frames):")
    print(f"    scene_type:      {final['scene_type']}")
    print(f"    confidence:      {final['confidence']:.3f}")
    print(f"    dominant_angles: {final['dominant_angles']} deg")
    print(f"    vector_count:    {final['vector_count']}")
    return final


# ─────────────────────────────────────────────────────────────
#  Test C: synthetic intersection (N-S + E-W streams)
# ─────────────────────────────────────────────────────────────

def test_synthetic_intersection(n_frames: int = 40) -> dict:
    print(f"\n{'='*60}")
    print("Test C: Synthetic intersection (N-S + E-W vehicle streams)")
    print(f"{'='*60}")

    clf = SceneClassifier(window_frames=20, max_match_dist_px=90)

    for frame_i in range(n_frames):
        boxes = []
        # N-S stream: vehicles moving downward (positive dy in image coords)
        for j in range(4):
            x = 200 + j * 130
            y = (60 + frame_i * 18 + j * 30) % 800
            boxes.append((x, y, x + 65, y + 42, "car", 0.82))
        # E-W stream: vehicles moving right
        for j in range(4):
            x = (50 + frame_i * 22 + j * 120) % 1800
            y = 350 + j * 80
            boxes.append((x, y, x + 65, y + 42, "car", 0.82))
        clf.update(boxes)

        if frame_i % 8 == 7:
            r = clf.classify()
            print(
                f"  frame {frame_i+1:2d} | vectors={r['vector_count']:3d} | "
                f"→ {r['scene_type']} (conf={r['confidence']:.2f})"
            )

    final = clf.classify()
    print(f"\n  FINAL ({n_frames} frames):")
    print(f"    scene_type:      {final['scene_type']}")
    print(f"    confidence:      {final['confidence']:.3f}")
    print(f"    dominant_angles: {final['dominant_angles']} deg")
    print(f"    vector_count:    {final['vector_count']}")
    return final


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = {}

    video_path = str(settings.video_source)
    if os.path.exists(video_path):
        results["A_real"] = test_real_video(video_path, n_frames=80)
    else:
        print(f"\nTest A SKIPPED — video not found: {video_path}")

    results["B_synthetic_flow"]  = test_synthetic_single_flow()
    results["C_synthetic_inter"] = test_synthetic_intersection()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        if r:
            status = "OK" if (
                (name.startswith("A") and r.get("scene_type") in ("intersection", "uncertain")) or
                (name.startswith("B") and r.get("scene_type") == "single_flow") or
                (name.startswith("C") and r.get("scene_type") == "intersection")
            ) else "UNEXPECTED"
            print(f"  {name:28s} → {r.get('scene_type','—'):15s} conf={r.get('confidence',0):.3f}  [{status}]")
