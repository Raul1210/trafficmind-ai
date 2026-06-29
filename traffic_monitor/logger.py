"""
Central logging configuration.

Call configure_logging() exactly once at application startup (inside create_app).
After that, every module obtains its own logger via get_logger(__name__).

The Python logging module is thread-safe; no additional synchronisation needed.
"""
from __future__ import annotations

import logging
from traffic_monitor.config import settings

_configured = False


def configure_logging() -> None:
    """
    Configure the root logger from settings.
    Idempotent — safe to call multiple times (only acts on the first call).
    """
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format=settings.log_format,
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger. Call as: logger = get_logger(__name__)"""
    return logging.getLogger(name)
