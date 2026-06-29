"""
Custom access-control decorators for the auth system.

Usage:
    from traffic_monitor.auth.decorators import admin_required

    @app.route('/admin/something')
    @admin_required          # must come AFTER @login_required (or handles it internally)
    def admin_only_view():
        ...
"""
from __future__ import annotations

from functools import wraps

from flask import abort
from flask_login import current_user


def admin_required(f):
    """
    Decorator that allows access only to authenticated users with role='admin'.
    Returns 403 for authenticated non-admins; redirects to login for anonymous
    users (Flask-Login's @login_required behaviour is inherited via current_user check).
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated:
            # Delegate to login_manager.unauthorized() — same as @login_required
            from flask_login import current_app
            from flask import redirect, url_for, request
            from traffic_monitor.auth import login_manager
            return login_manager.unauthorized()
        if current_user.role != "admin":
            abort(403)
        return f(*args, **kwargs)
    return decorated
