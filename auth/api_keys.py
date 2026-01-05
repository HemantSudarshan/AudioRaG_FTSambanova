"""
API Key Management

Generate, verify, and manage API keys for programmatic access.
"""

import logging
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple

from sqlalchemy.orm import Session

from auth.models import APIKey, APIKeyCreate, APIKeyResponse, User

logger = logging.getLogger(__name__)

# API key prefix for identification
API_KEY_PREFIX = "ar_"  # AudioRag
API_KEY_LENGTH = 32


def generate_api_key() -> Tuple[str, str]:
    """
    Generate a new API key.
    
    Returns:
        Tuple of (full_key, key_hash)
    """
    # Generate random bytes
    random_bytes = secrets.token_bytes(API_KEY_LENGTH)
    
    # Create the full key with prefix
    key_body = secrets.token_urlsafe(API_KEY_LENGTH)
    full_key = f"{API_KEY_PREFIX}{key_body}"
    
    # Hash the key for storage
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    
    return full_key, key_hash


def hash_api_key(api_key: str) -> str:
    """Hash an API key for comparison."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def create_api_key(
    db: Session,
    user: User,
    key_data: APIKeyCreate,
) -> APIKeyResponse:
    """
    Create a new API key for a user.
    
    Args:
        db: Database session
        user: User creating the key
        key_data: API key creation data
        
    Returns:
        APIKeyResponse with the full key (shown only once)
    """
    # Generate the key
    full_key, key_hash = generate_api_key()
    
    # Calculate expiry
    expires_at = None
    if key_data.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=key_data.expires_in_days)
    
    # Create the key record
    api_key = APIKey(
        key_hash=key_hash,
        name=key_data.name,
        prefix=full_key[:12],  # Store first 12 chars for identification
        user_id=user.id,
        expires_at=expires_at,
    )
    
    db.add(api_key)
    db.commit()
    db.refresh(api_key)
    
    logger.info(f"Created API key '{key_data.name}' for user {user.username}")
    
    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        prefix=api_key.prefix,
        key=full_key,  # Full key shown only on creation
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
    )


def verify_api_key(
    db: Session,
    api_key: str,
) -> Optional[User]:
    """
    Verify an API key and return the associated user.
    
    Args:
        db: Database session
        api_key: API key to verify
        
    Returns:
        User if valid, None otherwise
    """
    # Check prefix
    if not api_key.startswith(API_KEY_PREFIX):
        logger.warning("Invalid API key prefix")
        return None
    
    # Hash the key
    key_hash = hash_api_key(api_key)
    
    # Find the key in database
    key_record = db.query(APIKey).filter(
        APIKey.key_hash == key_hash,
        APIKey.is_active == True,
    ).first()
    
    if not key_record:
        logger.warning("API key not found or inactive")
        return None
    
    # Check expiry
    if key_record.expires_at and key_record.expires_at < datetime.utcnow():
        logger.warning(f"API key expired: {key_record.prefix}")
        return None
    
    # Check if user is active
    user = key_record.user
    if not user.is_active:
        logger.warning(f"API key user inactive: {user.username}")
        return None
    
    # Update last used timestamp
    key_record.last_used_at = datetime.utcnow()
    db.commit()
    
    logger.info(f"API key verified for user: {user.username}")
    return user


def revoke_api_key(
    db: Session,
    user: User,
    key_id: int,
) -> bool:
    """
    Revoke an API key.
    
    Args:
        db: Database session
        user: User revoking the key
        key_id: ID of the key to revoke
        
    Returns:
        True if revoked, False if not found
    """
    # Find the key
    key_record = db.query(APIKey).filter(
        APIKey.id == key_id,
        APIKey.user_id == user.id,
    ).first()
    
    if not key_record:
        logger.warning(f"API key {key_id} not found for user {user.username}")
        return False
    
    key_record.is_active = False
    db.commit()
    
    logger.info(f"Revoked API key {key_record.prefix} for user {user.username}")
    return True


def list_api_keys(
    db: Session,
    user: User,
    include_inactive: bool = False,
) -> list:
    """
    List all API keys for a user.
    
    Args:
        db: Database session
        user: User whose keys to list
        include_inactive: Include revoked keys
        
    Returns:
        List of API key records (without full keys)
    """
    query = db.query(APIKey).filter(APIKey.user_id == user.id)
    
    if not include_inactive:
        query = query.filter(APIKey.is_active == True)
    
    return query.order_by(APIKey.created_at.desc()).all()


def cleanup_expired_keys(db: Session) -> int:
    """
    Deactivate expired API keys.
    
    Args:
        db: Database session
        
    Returns:
        Number of keys deactivated
    """
    now = datetime.utcnow()
    
    result = db.query(APIKey).filter(
        APIKey.is_active == True,
        APIKey.expires_at != None,
        APIKey.expires_at < now,
    ).update({"is_active": False})
    
    db.commit()
    
    if result > 0:
        logger.info(f"Deactivated {result} expired API keys")
    
    return result
