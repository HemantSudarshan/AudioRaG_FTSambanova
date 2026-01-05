"""
Authorization Service

Role-based access control (RBAC) with decorators and middleware.
"""

import logging
from functools import wraps
from typing import Callable, List, Optional, Union

from auth.models import User, RoleType, PermissionType

logger = logging.getLogger(__name__)


class AuthorizationError(Exception):
    """Raised when authorization fails."""
    def __init__(self, message: str, required: str = None):
        self.message = message
        self.required = required
        super().__init__(message)


def has_role(user: User, role: Union[str, RoleType]) -> bool:
    """
    Check if a user has a specific role.
    
    Args:
        user: User object
        role: Role name or RoleType enum
        
    Returns:
        True if user has the role
    """
    role_name = role.value if isinstance(role, RoleType) else role
    return user.has_role(role_name)


def has_permission(user: User, permission: Union[str, PermissionType]) -> bool:
    """
    Check if a user has a specific permission.
    
    Args:
        user: User object
        permission: Permission name or PermissionType enum
        
    Returns:
        True if user has the permission
    """
    perm_name = permission.value if isinstance(permission, PermissionType) else permission
    
    # Admin has all permissions
    if user.has_role(RoleType.ADMIN.value):
        return True
    
    return user.has_permission(perm_name)


def has_any_role(user: User, roles: List[Union[str, RoleType]]) -> bool:
    """
    Check if a user has any of the specified roles.
    
    Args:
        user: User object
        roles: List of role names or RoleType enums
        
    Returns:
        True if user has any of the roles
    """
    return any(has_role(user, role) for role in roles)


def has_all_roles(user: User, roles: List[Union[str, RoleType]]) -> bool:
    """
    Check if a user has all of the specified roles.
    
    Args:
        user: User object
        roles: List of role names or RoleType enums
        
    Returns:
        True if user has all of the roles
    """
    return all(has_role(user, role) for role in roles)


def has_any_permission(user: User, permissions: List[Union[str, PermissionType]]) -> bool:
    """
    Check if a user has any of the specified permissions.
    
    Args:
        user: User object
        permissions: List of permission names or PermissionType enums
        
    Returns:
        True if user has any of the permissions
    """
    return any(has_permission(user, perm) for perm in permissions)


def require_role(
    roles: Union[str, RoleType, List[Union[str, RoleType]]],
    require_all: bool = False,
) -> Callable:
    """
    Decorator to require specific role(s) for a function.
    
    Args:
        roles: Required role(s)
        require_all: If True, user must have all roles
        
    Returns:
        Decorated function that checks roles
        
    Example:
        @require_role(RoleType.ADMIN)
        def admin_only_function(user, ...):
            ...
            
        @require_role([RoleType.ADMIN, RoleType.ANALYST])
        def analysts_or_admin_function(user, ...):
            ...
    """
    if not isinstance(roles, list):
        roles = [roles]
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(user: User, *args, **kwargs):
            if user is None:
                raise AuthorizationError(
                    "Authentication required",
                    required=str(roles)
                )
            
            if require_all:
                authorized = has_all_roles(user, roles)
            else:
                authorized = has_any_role(user, roles)
            
            if not authorized:
                role_names = [r.value if isinstance(r, RoleType) else r for r in roles]
                logger.warning(
                    f"User {user.username} denied access to {func.__name__}. "
                    f"Required roles: {role_names}"
                )
                raise AuthorizationError(
                    f"Access denied. Required role(s): {', '.join(role_names)}",
                    required=str(role_names)
                )
            
            return func(user, *args, **kwargs)
        return wrapper
    return decorator


def require_permission(
    permissions: Union[str, PermissionType, List[Union[str, PermissionType]]],
    require_all: bool = False,
) -> Callable:
    """
    Decorator to require specific permission(s) for a function.
    
    Args:
        permissions: Required permission(s)
        require_all: If True, user must have all permissions
        
    Returns:
        Decorated function that checks permissions
        
    Example:
        @require_permission(PermissionType.AUDIO_UPLOAD)
        def upload_audio(user, file):
            ...
    """
    if not isinstance(permissions, list):
        permissions = [permissions]
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(user: User, *args, **kwargs):
            if user is None:
                raise AuthorizationError(
                    "Authentication required",
                    required=str(permissions)
                )
            
            if require_all:
                authorized = all(has_permission(user, p) for p in permissions)
            else:
                authorized = has_any_permission(user, permissions)
            
            if not authorized:
                perm_names = [p.value if isinstance(p, PermissionType) else p for p in permissions]
                logger.warning(
                    f"User {user.username} denied access to {func.__name__}. "
                    f"Required permissions: {perm_names}"
                )
                raise AuthorizationError(
                    f"Access denied. Required permission(s): {', '.join(perm_names)}",
                    required=str(perm_names)
                )
            
            return func(user, *args, **kwargs)
        return wrapper
    return decorator


def is_admin(user: User) -> bool:
    """Check if user is an admin."""
    return has_role(user, RoleType.ADMIN)


def is_analyst(user: User) -> bool:
    """Check if user is an analyst."""
    return has_role(user, RoleType.ANALYST)


def is_viewer(user: User) -> bool:
    """Check if user is a viewer."""
    return has_role(user, RoleType.VIEWER)


def can_upload_audio(user: User) -> bool:
    """Check if user can upload audio."""
    return has_permission(user, PermissionType.AUDIO_UPLOAD)


def can_query(user: User) -> bool:
    """Check if user can create queries."""
    return has_permission(user, PermissionType.QUERY_CREATE)


def can_export_analytics(user: User) -> bool:
    """Check if user can export analytics."""
    return has_permission(user, PermissionType.ANALYTICS_EXPORT)


def can_manage_users(user: User) -> bool:
    """Check if user can manage other users."""
    return has_permission(user, PermissionType.USER_MANAGE)


# ===================================
# Streamlit Integration Helpers
# ===================================

def check_streamlit_auth(required_role: Optional[RoleType] = None) -> bool:
    """
    Check if current Streamlit session is authenticated.
    
    Use in Streamlit apps:
        if not check_streamlit_auth(RoleType.ANALYST):
            st.error("Access denied")
            st.stop()
    
    Args:
        required_role: Optional role requirement
        
    Returns:
        True if authenticated (and authorized if role specified)
    """
    try:
        import streamlit as st
        
        user = st.session_state.get("current_user")
        
        if not user:
            return False
        
        if required_role and not has_role(user, required_role):
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error checking Streamlit auth: {e}")
        return False
