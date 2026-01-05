"""
Authentication Service

JWT token generation, password hashing, and user authentication.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
import secrets

from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from auth.models import User, TokenPayload, TokenResponse, UserLogin

logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    user: User,
    secret_key: str,
    algorithm: str = "HS256",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token for a user.
    
    Args:
        user: User object
        secret_key: Secret key for signing
        algorithm: JWT algorithm
        expires_delta: Token expiry duration
        
    Returns:
        Encoded JWT token
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=60)
    
    now = datetime.utcnow()
    expire = now + expires_delta
    
    payload = {
        "sub": str(user.id),
        "exp": expire,
        "iat": now,
        "type": "access",
        "email": user.email,
        "username": user.username,
        "roles": [role.name for role in user.roles],
        "tenant_id": user.tenant_id,
    }
    
    token = jwt.encode(payload, secret_key, algorithm=algorithm)
    logger.info(f"Created access token for user {user.username}")
    return token


def create_refresh_token(
    user: User,
    secret_key: str,
    algorithm: str = "HS256",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT refresh token for a user.
    
    Args:
        user: User object
        secret_key: Secret key for signing
        algorithm: JWT algorithm
        expires_delta: Token expiry duration
        
    Returns:
        Encoded JWT refresh token
    """
    if expires_delta is None:
        expires_delta = timedelta(days=7)
    
    now = datetime.utcnow()
    expire = now + expires_delta
    
    payload = {
        "sub": str(user.id),
        "exp": expire,
        "iat": now,
        "type": "refresh",
        "jti": secrets.token_hex(16),  # Unique token ID
    }
    
    token = jwt.encode(payload, secret_key, algorithm=algorithm)
    logger.info(f"Created refresh token for user {user.username}")
    return token


def verify_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256",
    token_type: str = "access",
) -> Optional[TokenPayload]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        secret_key: Secret key for verification
        algorithm: JWT algorithm
        token_type: Expected token type ("access" or "refresh")
        
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        
        # Verify token type
        if payload.get("type") != token_type:
            logger.warning(f"Token type mismatch: expected {token_type}, got {payload.get('type')}")
            return None
        
        return TokenPayload(
            sub=payload["sub"],
            exp=datetime.fromtimestamp(payload["exp"]),
            iat=datetime.fromtimestamp(payload["iat"]),
            type=payload["type"],
            roles=payload.get("roles", []),
            tenant_id=payload.get("tenant_id"),
        )
        
    except JWTError as e:
        logger.warning(f"JWT verification failed: {e}")
        return None


def create_token_pair(
    user: User,
    secret_key: str,
    algorithm: str = "HS256",
    access_expires: Optional[timedelta] = None,
    refresh_expires: Optional[timedelta] = None,
) -> TokenResponse:
    """
    Create both access and refresh tokens for a user.
    
    Args:
        user: User object
        secret_key: Secret key for signing
        algorithm: JWT algorithm
        access_expires: Access token expiry
        refresh_expires: Refresh token expiry
        
    Returns:
        TokenResponse with both tokens
    """
    if access_expires is None:
        access_expires = timedelta(minutes=60)
    if refresh_expires is None:
        refresh_expires = timedelta(days=7)
    
    access_token = create_access_token(user, secret_key, algorithm, access_expires)
    refresh_token = create_refresh_token(user, secret_key, algorithm, refresh_expires)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=int(access_expires.total_seconds()),
    )


def authenticate_user(
    db: Session,
    credentials: UserLogin,
) -> Optional[User]:
    """
    Authenticate a user with username/email and password.
    
    Args:
        db: Database session
        credentials: Login credentials
        
    Returns:
        User object if authenticated, None otherwise
    """
    # Try to find user by username or email
    user = db.query(User).filter(
        (User.username == credentials.username) | (User.email == credentials.username)
    ).first()
    
    if not user:
        logger.warning(f"Login failed: user not found - {credentials.username}")
        return None
    
    if not verify_password(credentials.password, user.hashed_password):
        logger.warning(f"Login failed: invalid password - {credentials.username}")
        return None
    
    if not user.is_active:
        logger.warning(f"Login failed: user inactive - {credentials.username}")
        return None
    
    # Update last login
    user.last_login = datetime.utcnow()
    db.commit()
    
    logger.info(f"User authenticated: {user.username}")
    return user


def get_current_user(
    token: str,
    db: Session,
    secret_key: str,
    algorithm: str = "HS256",
) -> Optional[User]:
    """
    Get the current user from a JWT token.
    
    Args:
        token: JWT access token
        db: Database session
        secret_key: Secret key for verification
        algorithm: JWT algorithm
        
    Returns:
        User object if valid token, None otherwise
    """
    payload = verify_token(token, secret_key, algorithm, "access")
    
    if not payload:
        return None
    
    user = db.query(User).filter(User.id == int(payload.sub)).first()
    
    if not user or not user.is_active:
        return None
    
    return user


def refresh_access_token(
    refresh_token: str,
    db: Session,
    secret_key: str,
    algorithm: str = "HS256",
) -> Optional[str]:
    """
    Refresh an access token using a refresh token.
    
    Args:
        refresh_token: JWT refresh token
        db: Database session
        secret_key: Secret key
        algorithm: JWT algorithm
        
    Returns:
        New access token or None if refresh fails
    """
    payload = verify_token(refresh_token, secret_key, algorithm, "refresh")
    
    if not payload:
        return None
    
    user = db.query(User).filter(User.id == int(payload.sub)).first()
    
    if not user or not user.is_active:
        return None
    
    return create_access_token(user, secret_key, algorithm)
