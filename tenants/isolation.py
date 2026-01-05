"""
Tenant Isolation

Context management and middleware for multi-tenancy.
"""

import logging
from contextvars import ContextVar
from functools import wraps
from typing import Optional, Callable

from sqlalchemy.orm import Session

from tenants.models import Tenant, TenantPlan, PLAN_LIMITS

logger = logging.getLogger(__name__)

# Context variable for current tenant
_current_tenant: ContextVar[Optional[Tenant]] = ContextVar("current_tenant", default=None)


def get_tenant_context() -> Optional[Tenant]:
    """Get current tenant from context."""
    return _current_tenant.get()


def set_tenant_context(tenant: Optional[Tenant]):
    """Set current tenant in context."""
    _current_tenant.set(tenant)


def clear_tenant_context():
    """Clear tenant from context."""
    _current_tenant.set(None)


def tenant_required(func: Callable) -> Callable:
    """
    Decorator to require tenant context.
    
    Raises error if no tenant is set.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tenant = get_tenant_context()
        if not tenant:
            raise TenantError("Tenant context required")
        return func(*args, **kwargs)
    return wrapper


class TenantError(Exception):
    """Tenant-related error."""
    pass


class TenantMiddleware:
    """
    Middleware to set tenant context from request.
    
    Extracts tenant from:
    1. X-Tenant-ID header
    2. API key's associated tenant
    3. User's associated tenant
    """
    
    def __init__(self, db_session_factory):
        self.db_session_factory = db_session_factory
    
    def resolve_tenant(
        self,
        tenant_id: Optional[str] = None,
        api_key_tenant: Optional[str] = None,
        user_tenant: Optional[str] = None,
    ) -> Optional[Tenant]:
        """
        Resolve tenant from various sources.
        
        Priority: explicit > api_key > user
        """
        tenant_id = tenant_id or api_key_tenant or user_tenant
        
        if not tenant_id:
            return None
        
        with self.db_session_factory() as db:
            tenant = db.query(Tenant).filter(
                Tenant.id == tenant_id,
                Tenant.is_active == True,
            ).first()
            
            return tenant
    
    def __call__(self, request):
        """Process request and set tenant context."""
        # Extract tenant ID from header
        tenant_id = request.headers.get("X-Tenant-ID")
        
        tenant = self.resolve_tenant(tenant_id)
        
        if tenant:
            set_tenant_context(tenant)
            logger.debug(f"Tenant context set: {tenant.id}")
        
        return None  # Continue to next middleware


def get_tenant_collection_name(base_name: str) -> str:
    """
    Get tenant-specific collection name for Qdrant.
    
    Args:
        base_name: Base collection name
        
    Returns:
        Tenant-prefixed collection name
    """
    tenant = get_tenant_context()
    
    if tenant:
        prefix = tenant.collection_prefix or tenant.id
        return f"{prefix}_{base_name}"
    
    return base_name


def check_tenant_quota(
    tenant: Tenant,
    resource: str,
    amount: float = 1,
) -> bool:
    """
    Check if tenant has quota for resource.
    
    Args:
        tenant: Tenant to check
        resource: Resource type ("audio_hours", "storage_gb", "users")
        amount: Amount to check
        
    Returns:
        True if within quota
    """
    limits = PLAN_LIMITS.get(TenantPlan(tenant.plan), PLAN_LIMITS[TenantPlan.FREE])
    
    if resource == "audio_hours":
        max_limit = limits["max_audio_hours"]
        current = tenant.current_audio_hours or 0
        
        if max_limit == -1:  # Unlimited
            return True
        
        return (current + amount) <= max_limit
    
    elif resource == "storage_gb":
        max_limit = limits["max_storage_gb"]
        current_gb = (tenant.current_storage_used_mb or 0) / 1024
        
        if max_limit == -1:
            return True
        
        return (current_gb + amount) <= max_limit
    
    elif resource == "users":
        max_limit = limits["max_users"]
        
        if max_limit == -1:
            return True
        
        # Would need to count actual users
        return True
    
    return False


def has_feature(tenant: Tenant, feature: str) -> bool:
    """
    Check if tenant plan includes feature.
    
    Args:
        tenant: Tenant to check
        feature: Feature name
        
    Returns:
        True if feature available
    """
    limits = PLAN_LIMITS.get(TenantPlan(tenant.plan), PLAN_LIMITS[TenantPlan.FREE])
    features = limits.get("features", [])
    
    return feature in features
