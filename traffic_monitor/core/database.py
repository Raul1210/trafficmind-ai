"""
SQLite persistence layer.

Responsibilities:
  - init_db()       : create the schema once at startup (called by TrafficSystem)
  - DatabaseLogger  : background thread that writes one row every DB_LOGGING_INTERVAL
                      seconds using a snapshot provided by a callable.

Thread safety notes:
  - db_connected and db_row_count are only WRITTEN by the DatabaseLogger thread.
    They are READ by route handlers. Because Python's GIL makes bool and int reads
    atomic for CPython, no additional lock is needed for these two attributes.
  - The SQLite connection is owned exclusively by the DatabaseLogger thread —
    never shared.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from typing import Callable, List, Optional, Tuple

from traffic_monitor.config import Settings
from traffic_monitor.logger import get_logger

logger = get_logger(__name__)

# Snapshot type: (vehicles_count, avg_confidence, congestion_percentage)
SnapshotFn = Callable[[], Tuple[int, float, float]]


# ---------------------------------------------------------------------------
# Schema initialisation (runs once at startup, in the main thread)
# ---------------------------------------------------------------------------

def init_db(db_path: str, timeout: int = 10) -> bool:
    """
    Create the traffic_data table if it does not already exist.
    Also applies the congestion_percentage column migration for older databases.

    Returns:
        True on success, False if the database could not be opened or modified.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS traffic_data (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp             TEXT    NOT NULL,
                vehicles_count        INTEGER NOT NULL,
                avg_confidence        REAL    NOT NULL,
                traffic_intensity     REAL    NOT NULL,
                congestion_percentage REAL    NOT NULL DEFAULT 0
            )
        """)
        # Migration: add congestion_percentage to databases created before it existed
        try:
            conn.execute("SELECT congestion_percentage FROM traffic_data LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute(
                "ALTER TABLE traffic_data "
                "ADD COLUMN congestion_percentage REAL NOT NULL DEFAULT 0"
            )
            logger.info("Applied schema migration: added congestion_percentage column")
        conn.commit()
        conn.close()
        logger.info(f"Database ready: {db_path}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Database initialisation error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during DB init: {e}")
        return False


# ---------------------------------------------------------------------------
# Background writer thread
# ---------------------------------------------------------------------------

class DatabaseLogger(threading.Thread):
    """
    Runs in its own daemon thread and writes one traffic snapshot row
    to SQLite every cfg.db_logging_interval seconds.

    Public attributes (safe to read from any thread):
        db_connected : bool  — whether the last write succeeded
        db_row_count : int   — total rows in traffic_data
    """

    def __init__(self, cfg: Settings, get_snapshot: SnapshotFn, alert_manager=None) -> None:
        super().__init__(daemon=True, name="DatabaseLogger")
        self._cfg = cfg
        self._get_snapshot = get_snapshot
        self._alert_mgr = alert_manager
        self._stop = threading.Event()

        # Read by route handlers — written only by this thread (GIL-safe)
        self.db_connected: bool = False
        self.db_row_count: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the thread to exit its write loop and terminate."""
        self._stop.set()

    # ── Thread body ───────────────────────────────────────────────────────

    def run(self) -> None:
        cfg = self._cfg
        db_path = str(cfg.database_path)

        try:
            conn = sqlite3.connect(db_path, timeout=cfg.db_connection_timeout)
            self.db_connected = True
            logger.info("DatabaseLogger connected")
        except sqlite3.Error as e:
            logger.error(f"DatabaseLogger could not connect: {e}")
            return

        # Main write loop — wakes up every db_logging_interval seconds.
        # threading.Event.wait() is interruptible by stop(), unlike time.sleep().
        while not self._stop.wait(timeout=cfg.db_logging_interval):
            self._write_snapshot(conn, cfg)

        self._close(conn)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _write_snapshot(self, conn: sqlite3.Connection, cfg: Settings) -> None:
        vehicles, avg_conf, congestion = self._get_snapshot()
        traffic_intensity = vehicles / 10.0
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        inserted = False
        retry_count = 0

        while retry_count < cfg.db_max_retries and not inserted:
            try:
                conn.execute(
                    "INSERT INTO traffic_data "
                    "(timestamp, vehicles_count, avg_confidence, "
                    " traffic_intensity, congestion_percentage) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        timestamp,
                        vehicles,
                        round(avg_conf, 4),
                        round(traffic_intensity, 4),
                        round(congestion, 2),
                    ),
                )
                conn.commit()
                result = conn.execute("SELECT COUNT(*) FROM traffic_data").fetchone()
                self.db_row_count = result[0] if result else 0
                self.db_connected = True
                inserted = True
                logger.debug(
                    f"DB write: {vehicles} vehicles, {congestion:.1f}% congestion"
                )
                if self._alert_mgr:
                    try:
                        self._alert_mgr.check_and_alert(
                            congestion=congestion,
                            vehicles=vehicles,
                            avg_conf=avg_conf,
                            intensity=traffic_intensity,
                            video_src=str(self._cfg.video_source),
                        )
                    except Exception as e:
                        logger.warning(f'Alert check failed: {e}')

            except sqlite3.OperationalError as e:
                retry_count += 1
                logger.warning(
                    f"DB write attempt {retry_count}/{cfg.db_max_retries}: {e}"
                )
                if retry_count < cfg.db_max_retries:
                    time.sleep(cfg.db_retry_delay)
                else:
                    self.db_connected = False

            except sqlite3.DatabaseError as e:
                logger.error(f"DB error — attempting reconnect: {e}")
                conn = self._reconnect(conn, cfg)

            except Exception as e:
                logger.error(f"Unexpected DB write error: {e}")
                break

    def _reconnect(self, old_conn: sqlite3.Connection, cfg: Settings) -> sqlite3.Connection:
        try:
            old_conn.close()
        except Exception as e:
            logger.warning(f"Failed to close old DB connection: {e}")
        try:
            new_conn = sqlite3.connect(
                str(cfg.database_path), timeout=cfg.db_connection_timeout
            )
            self.db_connected = True
            logger.info("DatabaseLogger reconnected")
            return new_conn
        except Exception as e:
            logger.warning(f"DB reconnect failed: {e}")
            self.db_connected = False
            return old_conn  # caller will retry next interval

    def _close(self, conn: sqlite3.Connection) -> None:
        try:
            conn.close()
            logger.info("DatabaseLogger connection closed")
        except Exception as e:
            logger.warning(f"Failed to close DB connection: {e}")


# ---------------------------------------------------------------------------
# Users table — called once at startup alongside init_db()
# ---------------------------------------------------------------------------

def init_users_table(db_path: str, timeout: int = 10) -> bool:
    """
    Create the users table if it does not already exist.
    Safe to call on every startup (IF NOT EXISTS guard).
    """
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    UNIQUE NOT NULL,
                password_hash TEXT    NOT NULL,
                role          TEXT    NOT NULL CHECK(role IN ('admin', 'user')),
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login    TIMESTAMP,
                is_active     INTEGER DEFAULT 1
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Users table ready")
        return True
    except sqlite3.Error as e:
        logger.error(f"init_users_table error: {e}")
        return False


def create_default_admin(db_path: str, password_hash: str, timeout: int = 10) -> None:
    """
    Insert a default admin account (username='admin') only when the users
    table is completely empty.  Prints a visible console warning every time
    the default credentials are still in place.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count == 0:
            conn.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                ("admin", password_hash, "admin"),
            )
            conn.commit()
            logger.warning(
                "\n" + "=" * 62 + "\n"
                "  CONT ADMIN DEFAULT CREAT\n"
                "  username : admin\n"
                "  parola   : admin123\n"
                "  !! SCHIMBA PAROLA IMEDIAT dupa primul login!\n"
                "  Mergi la /admin/users pentru a modifica parola.\n"
                + "=" * 62
            )
        conn.close()
    except Exception as e:
        logger.error(f"create_default_admin error: {e}")


# ---------------------------------------------------------------------------
# Video sources table — tracks source switches for history panel
# ---------------------------------------------------------------------------

def init_video_sources_table(db_path: str, timeout: int = 10) -> bool:
    """Create the video_sources table if it does not already exist."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS video_sources (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type  TEXT      NOT NULL,
                source_value TEXT      NOT NULL,
                switched_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                switched_by  TEXT      NOT NULL
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Video sources table ready")
        return True
    except sqlite3.Error as e:
        logger.error(f"init_video_sources_table error: {e}")
        return False


def log_video_source_change(
    db_path: str,
    source_type: str,
    source_value: str,
    switched_by: str,
    timeout: int = 10,
) -> None:
    """Append one row to video_sources for audit / history panel."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.execute(
            "INSERT INTO video_sources (source_type, source_value, switched_by) "
            "VALUES (?, ?, ?)",
            (source_type, source_value, switched_by),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"log_video_source_change error: {e}")


# ---------------------------------------------------------------------------
# Email settings + alert history tables
# ---------------------------------------------------------------------------

def init_email_settings_table(db_path: str, timeout: int = 10) -> bool:
    """Create the email_settings singleton table if it does not already exist."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS email_settings (
                id                   INTEGER PRIMARY KEY CHECK (id = 1),
                smtp_server          TEXT    DEFAULT '',
                smtp_port            INTEGER DEFAULT 587,
                smtp_username        TEXT    DEFAULT '',
                smtp_password        TEXT    DEFAULT '',
                sender_email         TEXT    DEFAULT '',
                recipient_emails     TEXT    DEFAULT '',
                congestion_threshold INTEGER DEFAULT 85,
                consecutive_readings INTEGER DEFAULT 3,
                cooldown_minutes     INTEGER DEFAULT 5,
                enabled              INTEGER DEFAULT 0,
                updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("INSERT OR IGNORE INTO email_settings (id) VALUES (1)")
        conn.commit()
        conn.close()
        logger.info("Email settings table ready")
        return True
    except sqlite3.Error as e:
        logger.error(f"init_email_settings_table error: {e}")
        return False


def init_email_alert_history_table(db_path: str, timeout: int = 10) -> bool:
    """Create the email_alert_history table if it does not already exist."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS email_alert_history (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                sent_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                recipient_emails TEXT    NOT NULL,
                congestion_value REAL    NOT NULL,
                duration_seconds INTEGER NOT NULL,
                subject          TEXT    NOT NULL,
                status           TEXT    NOT NULL CHECK(status IN ('sent', 'failed')),
                error_message    TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Email alert history table ready")
        return True
    except sqlite3.Error as e:
        logger.error(f"init_email_alert_history_table error: {e}")
        return False


def get_email_settings(db_path: str, timeout: int = 10) -> Optional[dict]:
    """Return the email_settings row as a dict (includes plaintext password for internal use)."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM email_settings WHERE id = 1").fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.error(f"get_email_settings error: {e}")
        return None


def save_email_settings(db_path: str, settings_dict: dict, timeout: int = 10) -> bool:
    """
    Upsert the email_settings singleton row.
    If smtp_password is empty or all bullet characters, preserve the existing value.
    """
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.row_factory = sqlite3.Row

        incoming_pw = (settings_dict.get('smtp_password') or '').strip()
        if not incoming_pw or set(incoming_pw) == {'•'}:
            row = conn.execute(
                "SELECT smtp_password FROM email_settings WHERE id = 1"
            ).fetchone()
            smtp_password = row['smtp_password'] if row else ''
        else:
            smtp_password = incoming_pw

        conn.execute("""
            INSERT INTO email_settings
                (id, smtp_server, smtp_port, smtp_username, smtp_password,
                 sender_email, recipient_emails, congestion_threshold,
                 consecutive_readings, cooldown_minutes, enabled, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                smtp_server          = excluded.smtp_server,
                smtp_port            = excluded.smtp_port,
                smtp_username        = excluded.smtp_username,
                smtp_password        = excluded.smtp_password,
                sender_email         = excluded.sender_email,
                recipient_emails     = excluded.recipient_emails,
                congestion_threshold = excluded.congestion_threshold,
                consecutive_readings = excluded.consecutive_readings,
                cooldown_minutes     = excluded.cooldown_minutes,
                enabled              = excluded.enabled,
                updated_at           = excluded.updated_at
        """, (
            settings_dict.get('smtp_server', ''),
            int(settings_dict.get('smtp_port', 587)),
            settings_dict.get('smtp_username', ''),
            smtp_password,
            settings_dict.get('sender_email', ''),
            settings_dict.get('recipient_emails', ''),
            int(settings_dict.get('congestion_threshold', 85)),
            int(settings_dict.get('consecutive_readings', 3)),
            int(settings_dict.get('cooldown_minutes', 5)),
            int(bool(settings_dict.get('enabled', False))),
        ))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"save_email_settings error: {e}")
        return False


def get_email_alert_history(db_path: str, limit: int = 20, timeout: int = 10) -> List[dict]:
    """Return last `limit` email alert history rows, newest first."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, sent_at, recipient_emails, congestion_value, "
            "duration_seconds, subject, status, error_message "
            "FROM email_alert_history ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"get_email_alert_history error: {e}")
        return []
