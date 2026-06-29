"""
User model and database helpers for the auth system.

All reads/writes go directly to SQLite (same traffic_logs.db).
No ORM — keeps parity with the rest of the project's database layer.
"""
from __future__ import annotations

import sqlite3
from typing import Optional

from flask_login import UserMixin

from traffic_monitor.auth import login_manager
from traffic_monitor.config import settings
from traffic_monitor.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------

class User(UserMixin):
    def __init__(
        self,
        id: int,
        username: str,
        password_hash: str,
        role: str,
        is_active_flag: int,
    ) -> None:
        self.id = id
        self.username = username
        self.password_hash = password_hash
        self.role = role
        self._is_active = is_active_flag

    # Flask-Login expects `is_active` as a property
    @property
    def is_active(self) -> bool:
        return bool(self._is_active)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    def __repr__(self) -> str:  # pragma: no cover
        return f"User(id={self.id}, username={self.username!r}, role={self.role!r})"


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.database_path), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_user(row: sqlite3.Row) -> User:
    return User(
        id=row["id"],
        username=row["username"],
        password_hash=row["password_hash"],
        role=row["role"],
        is_active_flag=row["is_active"],
    )


def get_user_by_id(user_id: int) -> Optional[User]:
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT id, username, password_hash, role, is_active "
            "FROM users WHERE id = ?",
            (int(user_id),),
        ).fetchone()
        conn.close()
        return _row_to_user(row) if row else None
    except Exception as e:
        logger.error(f"get_user_by_id({user_id}): {e}")
        return None


def get_user_by_username(username: str) -> Optional[User]:
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT id, username, password_hash, role, is_active "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        conn.close()
        return _row_to_user(row) if row else None
    except Exception as e:
        logger.error(f"get_user_by_username({username!r}): {e}")
        return None


def get_all_users() -> list[dict]:
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT id, username, role, created_at, last_login, is_active "
            "FROM users ORDER BY id"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"get_all_users: {e}")
        return []


def update_last_login(user_id: int) -> None:
    try:
        conn = _connect()
        conn.execute(
            "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
            (user_id,),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"update_last_login({user_id}): {e}")


# ---------------------------------------------------------------------------
# Flask-Login user loader (registered on the login_manager singleton)
# ---------------------------------------------------------------------------

@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    return get_user_by_id(int(user_id))
