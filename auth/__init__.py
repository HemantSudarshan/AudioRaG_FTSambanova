"""
AudioRAG Authentication Module

Provides JWT-based authentication with role-based access control (RBAC).
"""

from auth.models import User, Role, Permission
from auth.authentication import (
    create_access_token,
    create_refresh_token,
    verify_token,
    get_password_hash,
    verify_password,
    get_current_user,
)
from auth.authorization import (
    require_role,
    require_permission,
    has_permission,
)
from auth.api_keys import (
    create_api_key,
    verify_api_key,
    revoke_api_key,
)

__all__ = [
    # Models
    "User",
    "Role", 
    "Permission",
    # Authentication
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "get_password_hash",
    "verify_password",
    "get_current_user",
    # Authorization
    "require_role",
    "require_permission",
    "has_permission",
    # API Keys
    "create_api_key",
    "verify_api_key",
    "revoke_api_key",
]
