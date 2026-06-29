"""
Auth Blueprint — login, logout, and admin user-management routes.

Routes registered (no url_prefix — paths are exactly as specified):
    GET  /login              Render login form
    POST /login              Process credentials
    GET  /logout             Clear session and redirect to /login
    GET  /admin/users        List all users  (admin only)
    POST /admin/users/create Create a new user  (admin only)
    POST /admin/users/<id>/toggle  Activate / deactivate a user  (admin only)
    POST /admin/users/<id>/delete  Delete a user  (admin only)
"""
from __future__ import annotations

import sqlite3

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_login import current_user, login_required, login_user, logout_user
from flask_wtf import FlaskForm
from wtforms import PasswordField, StringField, SubmitField
from wtforms.validators import DataRequired, Length

from traffic_monitor.auth import bcrypt
from traffic_monitor.auth.decorators import admin_required
from traffic_monitor.auth.models import (
    get_all_users,
    get_user_by_username,
    update_last_login,
)
from traffic_monitor.config import settings
from traffic_monitor.logger import get_logger

logger = get_logger(__name__)

auth_bp = Blueprint("auth", __name__)


# ---------------------------------------------------------------------------
# WTForms login form (CSRF protection via Flask-WTF)
# ---------------------------------------------------------------------------

class LoginForm(FlaskForm):
    username = StringField(
        "Username",
        validators=[DataRequired(message="Username is required."),
                    Length(min=2, max=80)],
    )
    password = PasswordField(
        "Password",
        validators=[DataRequired(message="Password is required.")],
    )
    submit = SubmitField("LOGIN")


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("api.index"))

    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data.strip()
        user = get_user_by_username(username)

        if (
            user
            and user.is_active
            and bcrypt.check_password_hash(user.password_hash, form.password.data)
        ):
            login_user(user, remember=False)
            update_last_login(user.id)
            logger.info(f"Login: '{user.username}' (role={user.role})")

            next_page = request.args.get("next", "")
            # Guard against open redirects — only allow relative paths
            if next_page and next_page.startswith("/") and not next_page.startswith("//"):
                return redirect(next_page)
            return redirect(url_for("api.index"))

        flash("Invalid credentials. Please try again.", "error")
        logger.warning(f"Failed login attempt for: {username!r}")

    return render_template("auth/login.html", form=form)


@auth_bp.route("/logout")
@login_required
def logout():
    username = current_user.username
    logout_user()
    logger.info(f"Logout: '{username}'")
    return redirect(url_for("auth.login"))


# ---------------------------------------------------------------------------
# Admin — user management
# ---------------------------------------------------------------------------

@auth_bp.route("/admin/users")
@admin_required
def admin_users():
    users = get_all_users()
    return render_template("admin/users.html", users=users)


@auth_bp.route("/admin/users/create", methods=["POST"])
@admin_required
def admin_create_user():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    role = request.form.get("role", "user")

    if not username or not password:
        flash("Username and password are required.", "error")
        return redirect(url_for("auth.admin_users"))

    if role not in ("admin", "user"):
        flash("Invalid role.", "error")
        return redirect(url_for("auth.admin_users"))

    if len(password) < 6:
        flash("Password must be at least 6 characters.", "error")
        return redirect(url_for("auth.admin_users"))

    pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")
    try:
        conn = sqlite3.connect(str(settings.database_path), timeout=10)
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, pw_hash, role),
        )
        conn.commit()
        conn.close()
        flash(f'Account "{username}" created successfully.', "success")
        logger.info(f"Admin created user '{username}' (role={role})")
    except sqlite3.IntegrityError:
        flash(f'Username "{username}" already exists.', "error")
    except Exception as e:
        flash(f"Error creating account: {e}", "error")
        logger.error(f"Error creating user '{username}': {e}")

    return redirect(url_for("auth.admin_users"))


@auth_bp.route("/admin/users/<int:user_id>/toggle", methods=["POST"])
@admin_required
def admin_toggle_user(user_id: int):
    if user_id == current_user.id:
        flash("You cannot deactivate your own account.", "error")
        return redirect(url_for("auth.admin_users"))
    try:
        conn = sqlite3.connect(str(settings.database_path), timeout=10)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT username, is_active FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row:
            new_state = 0 if row["is_active"] else 1
            conn.execute(
                "UPDATE users SET is_active = ? WHERE id = ?", (new_state, user_id)
            )
            conn.commit()
            state_str = "activated" if new_state else "deactivated"
            flash(f'Account "{row["username"]}" has been {state_str}.', "success")
        conn.close()
    except Exception as e:
        flash(f"Error: {e}", "error")
    return redirect(url_for("auth.admin_users"))


@auth_bp.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def admin_delete_user(user_id: int):
    if user_id == current_user.id:
        flash("You cannot delete your own account.", "error")
        return redirect(url_for("auth.admin_users"))
    try:
        conn = sqlite3.connect(str(settings.database_path), timeout=10)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT username FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if row:
            conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
            flash(f'Account "{row["username"]}" has been deleted.', "success")
        conn.close()
    except Exception as e:
        flash(f"Error deleting account: {e}", "error")
    return redirect(url_for("auth.admin_users"))
