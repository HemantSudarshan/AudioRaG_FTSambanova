"""
Audit Logger

Immutable audit trail for compliance and security.
"""

import json
import logging
import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from sqlalchemy import Column, String, Integer, DateTime, Text, Index
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base

logger = logging.getLogger(__name__)

Base = declarative_base()


class AuditAction(str, Enum):
    """Auditable actions."""
    # Authentication
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_LOGIN_FAILED = "user.login_failed"
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    PASSWORD_CHANGED = "user.password_changed"
    
    # API Keys
    API_KEY_CREATED = "api_key.created"
    API_KEY_REVOKED = "api_key.revoked"
    API_KEY_USED = "api_key.used"
    
    # Audio
    AUDIO_UPLOADED = "audio.uploaded"
    AUDIO_DELETED = "audio.deleted"
    AUDIO_TRANSCRIBED = "audio.transcribed"
    
    # Queries
    QUERY_EXECUTED = "query.executed"
    
    # Analytics
    ANALYTICS_VIEWED = "analytics.viewed"
    REPORT_EXPORTED = "analytics.exported"
    
    # Admin
    SETTINGS_CHANGED = "settings.changed"
    ROLE_ASSIGNED = "role.assigned"
    ROLE_REVOKED = "role.revoked"
    
    # Data
    DATA_EXPORTED = "data.exported"
    DATA_DELETED_GDPR = "data.deleted_gdpr"


class AuditLog(Base):
    """Audit log database model."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Who
    user_id = Column(String(50), index=True, nullable=True)
    username = Column(String(100), nullable=True)
    tenant_id = Column(String(50), index=True, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(String(500), nullable=True)
    
    # What
    action = Column(String(100), index=True, nullable=False)
    resource_type = Column(String(100), nullable=True)
    resource_id = Column(String(100), nullable=True)
    
    # Details
    details = Column(Text, nullable=True)  # JSON
    
    # When
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Integrity
    checksum = Column(String(64), nullable=False)  # SHA-256
    
    # Indexes for common queries
    __table_args__ = (
        Index('ix_audit_action_timestamp', 'action', 'timestamp'),
        Index('ix_audit_user_timestamp', 'user_id', 'timestamp'),
        Index('ix_audit_tenant_timestamp', 'tenant_id', 'timestamp'),
    )


@dataclass
class AuditEntry:
    """Audit entry data structure."""
    action: str
    user_id: Optional[str] = None
    username: Optional[str] = None
    tenant_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


class AuditLogger:
    """
    Audit logger for compliance tracking.
    
    Creates immutable, tamper-evident audit records.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self._file_logger = logging.getLogger("audit")
        self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Set up dedicated audit file logging."""
        if not self._file_logger.handlers:
            handler = logging.FileHandler("audit.log")
            handler.setFormatter(
                logging.Formatter('%(asctime)s - AUDIT - %(message)s')
            )
            self._file_logger.addHandler(handler)
            self._file_logger.setLevel(logging.INFO)
    
    def _compute_checksum(self, entry: AuditEntry) -> str:
        """
        Compute SHA-256 checksum for integrity verification.
        
        Args:
            entry: Audit entry
            
        Returns:
            Hex digest of checksum
        """
        data = json.dumps(asdict(entry), sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()
    
    def log(
        self,
        action: AuditAction,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        """
        Log an auditable action.
        
        Args:
            action: The action being logged
            user_id: ID of the user performing the action
            username: Username for display
            tenant_id: Tenant ID for multi-tenancy
            ip_address: Client IP address
            user_agent: Client user agent
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            details: Additional details as dict
            
        Returns:
            Created AuditLog record
        """
        timestamp = datetime.utcnow()
        
        entry = AuditEntry(
            action=action.value,
            user_id=user_id,
            username=username,
            tenant_id=tenant_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            timestamp=timestamp.isoformat(),
        )
        
        checksum = self._compute_checksum(entry)
        
        # Create database record
        log_entry = AuditLog(
            user_id=user_id,
            username=username,
            tenant_id=tenant_id,
            ip_address=ip_address,
            user_agent=user_agent,
            action=action.value,
            resource_type=resource_type,
            resource_id=resource_id,
            details=json.dumps(details) if details else None,
            timestamp=timestamp,
            checksum=checksum,
        )
        
        self.db.add(log_entry)
        self.db.commit()
        
        # Also log to file for redundancy
        self._file_logger.info(
            f"action={action.value} user={username or user_id} "
            f"resource={resource_type}:{resource_id} checksum={checksum[:8]}"
        )
        
        logger.debug(f"Audit logged: {action.value} by {username}")
        
        return log_entry
    
    def get_trail(
        self,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        action: Optional[AuditAction] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditLog]:
        """
        Retrieve audit trail with optional filters.
        
        Args:
            user_id: Filter by user
            tenant_id: Filter by tenant
            action: Filter by action type
            start_date: Start of date range
            end_date: End of date range
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of audit log entries
        """
        query = self.db.query(AuditLog)
        
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        
        if tenant_id:
            query = query.filter(AuditLog.tenant_id == tenant_id)
        
        if action:
            query = query.filter(AuditLog.action == action.value)
        
        if start_date:
            query = query.filter(AuditLog.timestamp >= start_date)
        
        if end_date:
            query = query.filter(AuditLog.timestamp <= end_date)
        
        return query.order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit).all()
    
    def verify_integrity(self, log_entry: AuditLog) -> bool:
        """
        Verify the integrity of an audit log entry.
        
        Args:
            log_entry: Audit log to verify
            
        Returns:
            True if checksum matches
        """
        details = json.loads(log_entry.details) if log_entry.details else None
        
        entry = AuditEntry(
            action=log_entry.action,
            user_id=log_entry.user_id,
            username=log_entry.username,
            tenant_id=log_entry.tenant_id,
            ip_address=log_entry.ip_address,
            user_agent=log_entry.user_agent,
            resource_type=log_entry.resource_type,
            resource_id=log_entry.resource_id,
            details=details,
            timestamp=log_entry.timestamp.isoformat(),
        )
        
        computed = self._compute_checksum(entry)
        return computed == log_entry.checksum


# ===================================
# Convenience Functions
# ===================================

_audit_logger: Optional[AuditLogger] = None


def init_audit_logger(db: Session):
    """Initialize the global audit logger."""
    global _audit_logger
    _audit_logger = AuditLogger(db)


def log_action(
    action: AuditAction,
    **kwargs
) -> Optional[AuditLog]:
    """
    Log an action using the global audit logger.
    
    Args:
        action: Action to log
        **kwargs: Additional audit parameters
        
    Returns:
        AuditLog entry or None if not initialized
    """
    if _audit_logger:
        return _audit_logger.log(action, **kwargs)
    else:
        logger.warning(f"Audit logger not initialized. Action not logged: {action.value}")
        return None


def get_audit_trail(**kwargs) -> List[AuditLog]:
    """
    Get audit trail using the global audit logger.
    
    Returns:
        List of audit entries
    """
    if _audit_logger:
        return _audit_logger.get_trail(**kwargs)
    return []
