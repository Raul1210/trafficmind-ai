"""
Auth package — Flask-Login + Flask-Bcrypt extension singletons.

Both are initialized lazily via init_app() in create_app(), so they can be
imported here without an active Flask application context.
"""
from flask_bcrypt import Bcrypt
from flask_login import LoginManager

login_manager = LoginManager()
login_manager.login_view = "auth.login"
login_manager.login_message = "You must be logged in to access this page."
login_manager.login_message_category = "warning"

bcrypt = Bcrypt()
