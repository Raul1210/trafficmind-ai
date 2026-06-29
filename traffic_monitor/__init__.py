"""
traffic_monitor — Flask application factory.

Usage:
    from traffic_monitor import create_app
    app = create_app()
    app.run(...)

The factory pattern:
  1. configure_logging() first (needed before any other module logs anything)
  2. validate_paths() — fast-fail if the YOLO model is missing
  3. Lazy-import TrafficSystem (this triggers YOLO model loading)
  4. Lazy-import register_routes
  5. Create system, register shutdown, register routes
  6. Return the Flask app
"""
from __future__ import annotations

import atexit
import sys

from flask import Flask, jsonify, render_template

from traffic_monitor.config import settings, validate_paths
from traffic_monitor.logger import configure_logging, get_logger


def create_app() -> Flask:
    """
    Application factory.  Call this exactly once per process.

    Returns:
        A fully configured Flask application with all routes registered
        and all background threads already running.
    """
    # ── 1. Logging must be configured before anything else logs ──────────
    configure_logging()
    logger = get_logger(__name__)

    # ── 2. Validate critical paths before loading the heavy model ─────────
    errors = validate_paths()
    if errors:
        for err in errors:
            logger.critical(err)
        sys.exit(1)

    # ── 3. Build the Flask app ────────────────────────────────────────────
    app = Flask(__name__, template_folder=str(settings.template_dir))
    app.config["SECRET_KEY"] = settings.secret_key
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload limit

    # Ensure the uploads directory exists at startup
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. CSRF protection ────────────────────────────────────────────────
    try:
        from flask_wtf.csrf import CSRFProtect
        CSRFProtect(app)
        logger.info("CSRF protection enabled")
    except ImportError:
        logger.warning(
            "flask-wtf not installed — CSRF protection disabled. "
            "Run: pip install flask-wtf"
        )

    # ── 5. Auth extensions (Flask-Login + Flask-Bcrypt) ───────────────────
    from traffic_monitor.auth import bcrypt, login_manager
    from traffic_monitor.auth.routes import auth_bp
    import traffic_monitor.auth.models  # registers @login_manager.user_loader

    login_manager.init_app(app)
    bcrypt.init_app(app)
    app.register_blueprint(auth_bp)
    logger.info("Auth blueprint registered (Flask-Login + Flask-Bcrypt)")

    # ── 6. Users table + default admin ────────────────────────────────────
    from traffic_monitor.core.database import (
        create_default_admin,
        init_email_alert_history_table,
        init_email_settings_table,
        init_users_table,
        init_video_sources_table,
    )

    init_users_table(str(settings.database_path))
    init_video_sources_table(str(settings.database_path))
    init_email_settings_table(str(settings.database_path))
    init_email_alert_history_table(str(settings.database_path))
    default_pw_hash = bcrypt.generate_password_hash("admin123").decode("utf-8")
    create_default_admin(str(settings.database_path), default_pw_hash)

    # ── 7. Create system (lazy import so YOLO loads AFTER validate_paths) ─
    from traffic_monitor.core.system import TrafficSystem
    from traffic_monitor.api.routes import register_routes

    system = TrafficSystem(settings)
    atexit.register(system.shutdown)

    # ── 8. Register all API/page routes ──────────────────────────────────
    register_routes(app, system, settings)

    # ── 9. Global error handlers ──────────────────────────────────────────
    @app.errorhandler(403)
    def forbidden(_e):
        return render_template("403.html"), 403

    logger.info(
        f"Application ready — "
        f"http://{settings.flask_host}:{settings.flask_port}"
    )
    return app
