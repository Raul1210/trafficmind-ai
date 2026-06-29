"""
Flask Blueprint — all HTTP routes.

Design principles:
  - Zero business logic here: routes delegate to system methods and return JSON/HTML.
  - No module-level import of TrafficSystem (injected via register_routes to avoid
    circular imports with traffic_monitor/__init__.py).
  - REST contract is IDENTICAL to the original app.py — every endpoint path,
    method, and response shape is preserved.
"""
from __future__ import annotations

import io
import csv
import json
import sqlite3
import time
from pathlib import Path

from flask import Blueprint, Response, jsonify, make_response, render_template, request
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename

from traffic_monitor.agent.q_learning import PHASE_NAMES, build_state
from traffic_monitor.auth.decorators import admin_required
from traffic_monitor.core.database import log_video_source_change
from traffic_monitor.logger import get_logger

logger = get_logger(__name__)

api_bp = Blueprint("api", __name__)

# Injected at startup via register_routes() — never imported at module level
_system = None
_cfg = None


def register_routes(app, system, cfg) -> None:
    """
    Bind the TrafficSystem and Settings to this Blueprint's handlers,
    then register the Blueprint with the Flask app.

    Called once inside create_app(), after the system is fully initialised.
    """
    global _system, _cfg
    _system = system
    _cfg = cfg
    app.register_blueprint(api_bp)


# ── Pages ──────────────────────────────────────────────────────────────────

@api_bp.route("/")
@login_required
def index():
    try:
        return render_template("index.html")
    except Exception as e:
        logger.error(f"Error rendering index: {e}")
        return jsonify({"error": "Failed to load dashboard"}), 500


@api_bp.route("/db_view")
@admin_required
def db_view():
    try:
        conn = sqlite3.connect(str(_cfg.database_path), timeout=10)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM traffic_data ORDER BY id DESC"
        ).fetchall()
        conn.close()
        data = [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"db_view error: {e}")
        data = []
    try:
        return render_template("db_view.html", data=data)
    except Exception as e:
        logger.error(f"Error rendering db_view: {e}")
        return jsonify({"error": "Failed to load database view"}), 500


# ── Video streaming ────────────────────────────────────────────────────────

@api_bp.route("/video_feed")
@login_required
def video_feed():
    def generate():
        frame_errors = 0
        max_errors = 300  # Large: gives yt-dlp time to initialise

        while True:
            try:
                frame_bytes = _system.capture.get_frame_bytes()
                if frame_bytes:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + frame_bytes
                        + b"\r\n"
                    )
                    frame_errors = 0
                else:
                    frame_errors += 1
                    if frame_errors % 50 == 0:
                        logger.warning(
                            f"Waiting for video stream... ({frame_errors} attempts)"
                        )
                    if frame_errors > max_errors:
                        frame_errors = 0  # Reset instead of break — generator never stops
                    time.sleep(0.1)

                time.sleep(0.033)  # ~30 FPS cap for streaming

            except Exception as e:
                logger.error(f"Video feed error: {e}")
                frame_errors += 1
                time.sleep(0.1)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── Stats & status ─────────────────────────────────────────────────────────

@api_bp.route("/stats")
@login_required
def get_stats():
    try:
        smoothed, _, congestion = _system.capture.get_snapshot()
        smoothed["congestion"] = round(congestion, 1)
        response = jsonify(smoothed)
        response.headers.update({
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma":        "no-cache",
            "Expires":       "0",
        })
        return response
    except Exception as e:
        logger.error(f"/stats error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/db_status")
@login_required
def db_status():
    try:
        return jsonify({
            "connected": _system.db_connected,
            "row_count": _system.db_row_count,
        })
    except Exception as e:
        logger.error(f"/db_status error: {e}")
        return jsonify({"connected": False, "row_count": 0, "error": str(e)}), 500


# ── Data exports ───────────────────────────────────────────────────────────

@api_bp.route("/export/json")
@admin_required
def export_json():
    try:
        conn = sqlite3.connect(str(_cfg.database_path), timeout=10)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM traffic_data ORDER BY id").fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    except Exception as e:
        logger.error(f"/export/json error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/export/csv")
@admin_required
def export_csv():
    try:
        conn = sqlite3.connect(str(_cfg.database_path), timeout=10)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM traffic_data ORDER BY id").fetchall()
        conn.close()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "id", "timestamp", "vehicles_count",
            "avg_confidence", "traffic_intensity", "congestion_percentage",
        ])
        for row in rows:
            writer.writerow([
                row["id"], row["timestamp"], row["vehicles_count"],
                row["avg_confidence"], row["traffic_intensity"],
                row["congestion_percentage"],
            ])
        response = make_response(output.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=traffic_data.csv"
        response.headers["Content-Type"] = "text/csv"
        return response
    except Exception as e:
        logger.error(f"/export/csv error: {e}")
        return jsonify({"error": str(e)}), 500


# ── RL / Q-agent API ───────────────────────────────────────────────────────

@api_bp.route("/api/counts")
@login_required
def api_counts():
    try:
        directional = _system.capture.get_directional_counts()
        smoothed, _, congestion = _system.capture.get_snapshot()
        return jsonify({
            **directional,
            "total_vehicles": smoothed.get("car", 0) + smoothed.get("motorbike", 0),
            "pedestrians":    smoothed.get("person", 0),
            "congestion":     round(congestion, 1),
        })
    except Exception as e:
        logger.error(f"/api/counts error: {e}")
        return jsonify({"north": 0, "south": 0, "east": 0, "west": 0}), 500


@api_bp.route("/api/signal")
@login_required
def api_signal():
    try:
        directional = _system.capture.get_directional_counts()
        congestion  = _system.capture.get_congestion()

        with _system.rl_lock:
            rl_active = _system.rl_state.get("rl_active", True)

            total_q    = sum(directional.values())
            prev_total = sum(_system.rl_state.get("queues", {}).values())
            reward     = -1.0 * total_q + 2.0 * max(0, prev_total - total_q)

            if rl_active:
                state  = build_state(
                    directional["north"], directional["south"],
                    directional["east"],  directional["west"],
                    _system.rl_agent.current_phase,
                )
                action = _system.rl_agent.choose_action(state)
                q_vals = _system.rl_agent.q_table[state].tolist()

                if _system.rl_agent.should_switch_phase(action):
                    _system.rl_agent.current_phase = action
                    _system.rl_agent.phase_duration = 0
                else:
                    _system.rl_agent.phase_duration += 1
            else:
                # Single-flow road: freeze RL state, don't advance the agent
                q_vals = _system.rl_state.get("q_values", [0.0, 0.0, 0.0, 0.0])

            _system.rl_state.update({
                "phase":           _system.rl_agent.current_phase,
                "phase_name":      PHASE_NAMES[_system.rl_agent.current_phase],
                "phase_duration":  _system.rl_agent.phase_duration,
                "q_values":        [round(v, 3) for v in q_vals],
                "confidence":      round(float(max(q_vals) - min(q_vals)), 3) if q_vals else 0.0,
                "queues":          directional,
                "reward":          round(reward, 2),
                "last_updated":    time.time(),
                "epsilon":         round(_system.rl_agent.epsilon, 3),
                "states_explored": len(_system.rl_agent.q_table),
                "congestion":      round(congestion, 1),
            })

        return jsonify(_system.rl_state)

    except Exception as e:
        logger.error(f"/api/signal error: {e}")
        return jsonify({
            "phase": 0, "phase_name": "N-S Green / E-W Red", "error": str(e)
        }), 500


@api_bp.route("/api/benchmark")
@login_required
def api_benchmark():
    try:
        with open(str(_cfg.benchmark_results_path)) as f:
            return jsonify(json.load(f))
    except FileNotFoundError:
        return jsonify({
            "error": "Run first: python -m traffic_monitor.agent.q_learning --mode benchmark",
            "low":    {"rl_avg_wait": 0, "fixed_avg_wait": 0, "wait_improvement_pct": 0},
            "medium": {"rl_avg_wait": 0, "fixed_avg_wait": 0, "wait_improvement_pct": 0},
            "peak":   {"rl_avg_wait": 0, "fixed_avg_wait": 0, "wait_improvement_pct": 0},
        })
    except Exception as e:
        logger.error(f"/api/benchmark error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/training")
@login_required
def api_training():
    try:
        with _system.rl_lock:
            q_values_per_phase = {}
            if _system.rl_agent and _system.rl_agent.q_table:
                table = _system.rl_agent.q_table
                n_actions = 4
                for action in range(n_actions):
                    vals = [v[action] for v in table.values() if len(v) > action]
                    q_values_per_phase[f"phase_{action}"] = float(max(vals)) if vals else 0.0
            return jsonify({
                "episode_rewards":    _system.rl_agent.episode_rewards[-200:],
                "episode_wait_times": _system.rl_agent.episode_wait_times[-200:],
                "episodes":           len(_system.rl_agent.episode_rewards),
                "epsilon":            round(_system.rl_agent.epsilon, 3),
                "states_explored":    len(_system.rl_agent.q_table),
                "q_values":           q_values_per_phase,
            })
    except Exception as e:
        logger.error(f"/api/training error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Video source management (admin only) ───────────────────────────────────

_ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _detect_source_type(value: str) -> str:
    """Infer source type from a string value."""
    v = str(value).strip()
    if v.isdigit():
        return "webcam"
    if "youtube.com" in v or "youtu.be" in v:
        return "youtube"
    return "file"


@api_bp.route("/api/video-source")
@login_required
def api_video_source():
    """Return the current active video source and recent history (last 5)."""
    try:
        current = str(_cfg.video_source)
        source_type = _detect_source_type(current)

        if source_type == "webcam":
            display = f"Webcam #{current}"
        elif source_type == "youtube":
            display = current
        else:
            display = Path(current).name

        # Recent source history
        try:
            conn = sqlite3.connect(str(_cfg.database_path), timeout=10)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT source_type, source_value, switched_at, switched_by "
                "FROM video_sources ORDER BY switched_at DESC LIMIT 5"
            ).fetchall()
            conn.close()
            recent = [dict(r) for r in rows]
        except Exception:
            recent = []

        return jsonify({
            "source_type": source_type,
            "source_value": current,
            "display_name": display,
            "recent": recent,
        })
    except Exception as e:
        logger.error(f"/api/video-source error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/video-source/change", methods=["POST"])
@admin_required
def api_change_video_source():
    """
    Switch the live video source at runtime.

    Request JSON:
        {
          "source_type":    "file"|"youtube"|"webcam",
          "source_value":   "...",
          "inference_size": 1280   (optional — use larger values for aerial sources)
        }
    Response JSON: {"ok": true, "source_type": "...", "source_value": "...", "inference_size": N}
    """
    data = request.get_json(silent=True) or {}
    source_type = str(data.get("source_type", "")).strip()
    source_value = str(data.get("source_value", "")).strip()

    # Optional inference_size override (e.g. 1280 for aerial/drone footage)
    raw_imgsz = data.get("inference_size")
    inference_size = None
    if raw_imgsz is not None:
        try:
            inference_size = int(raw_imgsz)
            if inference_size not in (320, 480, 640, 960, 1280, 1920):
                return jsonify({"ok": False, "error": "inference_size must be one of: 320, 480, 640, 960, 1280, 1920"}), 400
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "inference_size must be an integer"}), 400

    if source_type not in ("file", "youtube", "webcam"):
        return jsonify({"ok": False, "error": "Invalid source_type. Use: file, youtube, webcam"}), 400

    if not source_value:
        return jsonify({"ok": False, "error": "source_value is required"}), 400

    if source_type == "webcam":
        if not source_value.isdigit():
            return jsonify({"ok": False, "error": "Webcam source_value must be a camera index (integer)"}), 400
        new_source: str | int = int(source_value)
    elif source_type == "file":
        if not Path(source_value).exists():
            return jsonify({"ok": False, "error": f"File not found: {source_value}"}), 400
        new_source = source_value
    else:  # youtube
        if not ("youtube.com" in source_value or "youtu.be" in source_value):
            return jsonify({"ok": False, "error": "Not a valid YouTube URL"}), 400
        new_source = source_value

    try:
        _system.change_video_source(new_source, inference_size=inference_size)
        log_video_source_change(
            str(_cfg.database_path), source_type, str(source_value), current_user.username
        )
        logger.info(f"Video source switched to {source_type}:{source_value!r} by {current_user.username!r}"
                    + (f" inference_size={inference_size}" if inference_size else ""))
        return jsonify({
            "ok": True,
            "source_type": source_type,
            "source_value": str(source_value),
            "inference_size": inference_size,
        })
    except Exception as e:
        logger.error(f"/api/video-source/change error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@api_bp.route("/api/video-source/upload", methods=["POST"])
@admin_required
def api_upload_video():
    """
    Upload a video file and immediately switch the capture worker to it.

    Accepts: multipart/form-data with field name "file"
    Allowed extensions: .mp4 .avi .mov .mkv .webm
    Max size: controlled by Flask MAX_CONTENT_LENGTH (500 MB)
    """
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file field in request"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "No file selected"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in _ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({
            "ok": False,
            "error": f"Unsupported format '{ext}'. Allowed: {', '.join(_ALLOWED_VIDEO_EXTENSIONS)}"
        }), 400

    filename = secure_filename(f.filename)
    uploads_dir = _cfg.uploads_dir
    uploads_dir.mkdir(parents=True, exist_ok=True)
    save_path = uploads_dir / filename

    try:
        f.save(str(save_path))
        logger.info(f"Video uploaded: {save_path} by {current_user.username!r}")
    except Exception as e:
        logger.error(f"File save error: {e}")
        return jsonify({"ok": False, "error": f"Failed to save file: {e}"}), 500

    # Optional inference_size from form field (aerial uploads may need 1280)
    raw_imgsz = request.form.get("inference_size")
    inference_size = None
    if raw_imgsz is not None:
        try:
            inference_size = int(raw_imgsz)
        except (TypeError, ValueError):
            pass

    try:
        _system.change_video_source(str(save_path), inference_size=inference_size)
        log_video_source_change(
            str(_cfg.database_path), "file", str(save_path), current_user.username
        )
        return jsonify({
            "ok": True,
            "source_type": "file",
            "source_value": filename,
            "inference_size": inference_size,
        })
    except Exception as e:
        logger.error(f"/api/video-source/upload switch error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


# ── PDF session report ────────────────────────────────────────────────────

@api_bp.route("/api/report/pdf")
@login_required
@admin_required
def api_report_pdf():
    """
    Generate and download a PDF session report.

    Calls PDFReportGenerator.generate() which builds the document entirely
    in-memory (no temp files) and returns a BytesIO buffer.
    """
    try:
        from traffic_monitor.reports.pdf_generator import PDFReportGenerator
        gen = PDFReportGenerator(
            system=_system,
            db_path=str(_cfg.database_path),
            username=current_user.username,
            cfg=_cfg,
            start_time=_system.start_time,
        )
        pdf_buf = gen.generate()
        ts = time.strftime("%Y%m%d_%H%M%S")
        resp = make_response(pdf_buf.read())
        resp.headers["Content-Type"] = "application/pdf"
        resp.headers["Content-Disposition"] = (
            f'attachment; filename="trafficmind_report_{ts}.pdf"'
        )
        resp.headers["Cache-Control"] = "no-store"
        return resp
    except Exception as e:
        logger.error(f"/api/report/pdf error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Email alert settings ──────────────────────────────────────────────────

@api_bp.route("/api/email-settings")
@login_required
@admin_required
def api_get_email_settings():
    """Return current email settings with password masked (empty string)."""
    try:
        from traffic_monitor.core.database import get_email_settings
        cfg = get_email_settings(str(_cfg.database_path))
        if not cfg:
            return jsonify({"error": "No settings found"}), 404
        cfg['smtp_password'] = ''
        return jsonify(cfg)
    except Exception as e:
        logger.error(f"/api/email-settings GET error: {e}")
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/email-settings", methods=["POST"])
@login_required
@admin_required
def api_save_email_settings():
    """Save email alert settings. Empty/masked smtp_password preserves existing."""
    try:
        from traffic_monitor.core.database import save_email_settings
        data = request.get_json(silent=True) or {}
        ok = save_email_settings(str(_cfg.database_path), data)
        if ok:
            _system.email_alerts.invalidate_cache()
        return jsonify({"ok": ok})
    except Exception as e:
        logger.error(f"/api/email-settings POST error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@api_bp.route("/api/email-settings/test", methods=["POST"])
@login_required
@admin_required
def api_test_email():
    """Send a test email using the current SMTP configuration."""
    try:
        ok, err = _system.email_alerts.send_test_email()
        return jsonify({"ok": ok, "error": err})
    except Exception as e:
        logger.error(f"/api/email-settings/test error: {e}")
        return jsonify({"ok": False, "error": str(e)}), 500


@api_bp.route("/api/email-alert-history")
@login_required
@admin_required
def api_email_alert_history():
    """Return last 20 email alert history rows."""
    try:
        from traffic_monitor.core.database import get_email_alert_history
        alerts = get_email_alert_history(str(_cfg.database_path), limit=20)
        return jsonify({"alerts": alerts})
    except Exception as e:
        logger.error(f"/api/email-alert-history error: {e}")
        return jsonify({"alerts": [], "error": str(e)}), 500


# ── Error handlers ─────────────────────────────────────────────────────────

@api_bp.app_errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Resource not found"}), 404


@api_bp.app_errorhandler(500)
def internal_error(_e):
    return jsonify({"error": "Internal server error"}), 500
