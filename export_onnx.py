"""
Export YOLOv8s to optimized ONNX for CPU inference.

YOLOv8s ("small") offers significantly better detection accuracy than nano
while staying much faster than medium — a good middle ground at 15-20 FPS on CPU.

Run once from the project root:
    python export_onnx.py

Produces yolov8s.onnx in the workspace directory (next to yolov8m.pt).
After export, the app automatically prefers the ONNX model on next startup.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

# Workspace root: two levels above this script (where model .pt files live)
SCRIPT_DIR    = Path(__file__).parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent
SMALL_PT      = WORKSPACE_DIR / "yolov8s.pt"
SMALL_ONNX    = WORKSPACE_DIR / "yolov8s.onnx"


def main() -> None:
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed — run: pip install ultralytics")

    if SMALL_ONNX.exists():
        print(f"[OK] ONNX already exists: {SMALL_ONNX}")
        return

    print("Loading YOLOv8s (auto-download if needed)...")
    # YOLO("yolov8s.pt") auto-downloads small weights (~22 MB) on first use
    model = YOLO(str(SMALL_PT) if SMALL_PT.exists() else "yolov8s.pt")

    print("Exporting to ONNX (imgsz=480, opset=12, simplify=True)...")
    export_path = model.export(
        format="onnx",
        imgsz=480,
        opset=12,
        simplify=True,
    )
    print(f"Exported: {export_path}")

    # Move to workspace root if ultralytics saved it elsewhere
    exported = Path(export_path)
    if exported.resolve() != SMALL_ONNX.resolve():
        shutil.move(str(exported), str(SMALL_ONNX))
        print(f"Moved to: {SMALL_ONNX}")

    print("\n[DONE] Restart the Flask app — it will auto-load yolov8s.onnx.")


if __name__ == "__main__":
    main()
