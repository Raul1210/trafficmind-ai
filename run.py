"""
Entry point for the Smart Traffic Management System.

Run with:
    python run.py

Or with gunicorn for production:
    gunicorn "traffic_monitor:create_app()" --bind 0.0.0.0:5000 --workers 1

Note: use --workers 1 with gunicorn — the background threads (capture + DB)
must not be forked across multiple workers.
"""
from traffic_monitor import create_app
from traffic_monitor.config import settings
from traffic_monitor.logger import get_logger

app = create_app()

if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("Smart Traffic Management System — Starting")
    logger.info(f"Dashboard:  http://{settings.flask_host}:{settings.flask_port}/")
    logger.info(f"API signal: http://{settings.flask_host}:{settings.flask_port}/api/signal")
    logger.info("=" * 60)
    try:
        app.run(
            host=settings.flask_host,
            port=settings.flask_port,
            debug=settings.flask_debug,
            threaded=settings.flask_threaded,
        )
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal startup error: {e}")
        raise
