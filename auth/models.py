"""
Authentication Models

SQLAlchemy models for users, roles, and permissions.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, ForeignKey, Table, Text
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, EmailStr, Field

Base = declarative_base()


# ===================================
# Enums
# ===================================

class RoleType(str, Enum):
    """Available user roles."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class PermissionType(str, Enum):
    """Available permissions."""
    # Audio permissions
    AUDIO_UPLOAD = "audio:upload"
    AUDIO_READ = "audio:read"
    AUDIO_DELETE = "audio:delete"
    
    # Query permissions
    QUERY_CREATE = "query:create"
    QUERY_READ = "query:read"
    
    # Analytics permissions
    ANALYTICS_VIEW = "analytics:view"
    ANALYTICS_EXPORT = "analytics:export"
    
    # User management
    USER_MANAGE = "user:manage"
    USER_READ = "user:read"
    
    # Admin
    ADMIN_ALL = "admin:all"


# ===================================
# Association Tables
# ===================================

user_roles = Table(
    "user_roles",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True),
)

role_permissions = Table(
    "role_permissions",
    Base.metadata,
    Column("role_id", Integer, ForeignKey("roles.id"), primary_key=True),
    Column("permission_id", Integer, ForeignKey("permissions.id"), primary_key=True),
)


# ===================================
# SQLAlchemy Models
# ===================================

class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Tenant (for multi-tenancy)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(50), index=True, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    
    def has_role(self, role_name: str) -> bool:
        """Check if user has a specific role."""
        return any(role.name == role_name for role in self.roles)
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has a specific permission."""
        for role in self.roles:
            if role.name == RoleType.ADMIN:
                return True  # Admin has all permissions
            for perm in role.permissions:
                if perm.name == permission:
                    return True
        return False


class Role(Base):
    """Role model for RBAC."""
    __tablename__ = "roles"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    permissions = relationship("Permission", secondary=role_permissions, back_populates="roles")


class Permission(Base):
    """Permission model for fine-grained access control."""
    __tablename__ = "permissions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    roles = relationship("Role", secondary=role_permissions, back_populates="permissions")


class APIKey(Base):
    """API Key model for programmatic access."""
    __tablename__ = "api_keys"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    prefix: Mapped[str] = mapped_column(String(10), nullable=False)  # First 8 chars for identification
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Expiry
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Owner
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="api_keys")


# ===================================
# Pydantic Schemas
# ===================================

class UserCreate(BaseModel):
    """Schema for creating a new user."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    """Schema for user response (no password)."""
    id: int
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    is_verified: bool
    tenant_id: Optional[str]
    roles: List[str] = []
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """Schema for user login."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPayload(BaseModel):
    """Schema for decoded JWT payload."""
    sub: str  # User ID
    exp: datetime
    iat: datetime
    type: str  # "access" or "refresh"
    roles: List[str] = []
    tenant_id: Optional[str] = None


class APIKeyCreate(BaseModel):
    """Schema for creating an API key."""
    name: str = Field(..., min_length=1, max_length=100)
    expires_in_days: Optional[int] = Field(default=365, ge=1, le=3650)


class APIKeyResponse(BaseModel):
    """Schema for API key response (shown only once)."""
    id: int
    name: str
    prefix: str
    key: str  # Full key, shown only on creation
    expires_at: Optional[datetime]
    created_at: datetime


# ===================================
# Default Roles and Permissions
# ===================================

DEFAULT_PERMISSIONS = {
    RoleType.ADMIN: [
        PermissionType.ADMIN_ALL,
    ],
    RoleType.ANALYST: [
        PermissionType.AUDIO_UPLOAD,
        PermissionType.AUDIO_READ,
        PermissionType.AUDIO_DELETE,
        PermissionType.QUERY_CREATE,
        PermissionType.QUERY_READ,
        PermissionType.ANALYTICS_VIEW,
        PermissionType.ANALYTICS_EXPORT,
    ],
    RoleType.VIEWER: [
        PermissionType.AUDIO_READ,
        PermissionType.QUERY_READ,
        PermissionType.ANALYTICS_VIEW,
    ],
}
