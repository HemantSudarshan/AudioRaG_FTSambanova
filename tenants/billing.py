"""
Billing and Usage Tracking

Track tenant resource usage for billing.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

from sqlalchemy.orm import Session

from tenants.models import Tenant

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """Single usage record."""
    tenant_id: str
    resource_type: str  # audio_hours, storage_mb, queries, api_calls
    amount: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class UsageSummary:
    """Usage summary for billing."""
    tenant_id: str
    period_start: datetime
    period_end: datetime
    audio_hours: float
    storage_used_mb: float
    query_count: int
    api_call_count: int
    estimated_cost: float


class UsageTracker:
    """
    Track resource usage per tenant.
    
    Supports Redis for persistence and in-memory fallback.
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._memory_store: Dict[str, list] = defaultdict(list)
        
        # Pricing (per unit)
        self.pricing = {
            "audio_hours": 0.50,  # $0.50 per hour
            "storage_gb": 0.10,   # $0.10 per GB per month
            "queries": 0.001,    # $0.001 per query
            "api_calls": 0.0001, # $0.0001 per API call
        }
    
    def record(
        self,
        tenant_id: str,
        resource_type: str,
        amount: float,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record resource usage.
        
        Args:
            tenant_id: Tenant ID
            resource_type: Type of resource
            amount: Amount used
            metadata: Additional data
        """
        record = UsageRecord(
            tenant_id=tenant_id,
            resource_type=resource_type,
            amount=amount,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
        
        key = f"usage:{tenant_id}"
        
        if self.redis:
            try:
                import json
                self.redis.lpush(key, json.dumps(asdict(record), default=str))
                self.redis.expire(key, 86400 * 90)  # 90 days retention
            except Exception as e:
                logger.warning(f"Redis usage record failed: {e}")
                self._memory_store[key].append(record)
        else:
            self._memory_store[key].append(record)
        
        logger.debug(f"Usage recorded: {tenant_id} - {resource_type}: {amount}")
    
    def track_audio_upload(
        self,
        tenant_id: str,
        duration_seconds: float,
        file_size_mb: float,
    ):
        """Track audio upload usage."""
        hours = duration_seconds / 3600
        self.record(tenant_id, "audio_hours", hours, {
            "duration_seconds": duration_seconds,
            "file_size_mb": file_size_mb,
        })
        self.record(tenant_id, "storage_mb", file_size_mb)
    
    def track_query(self, tenant_id: str, latency_ms: float = 0):
        """Track query usage."""
        self.record(tenant_id, "queries", 1, {"latency_ms": latency_ms})
    
    def track_api_call(self, tenant_id: str, endpoint: str):
        """Track API call usage."""
        self.record(tenant_id, "api_calls", 1, {"endpoint": endpoint})
    
    def get_usage(
        self,
        tenant_id: str,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> float:
        """
        Get total usage for a resource.
        
        Args:
            tenant_id: Tenant ID
            resource_type: Filter by resource type
            start_date: Start of period
            end_date: End of period
            
        Returns:
            Total usage amount
        """
        key = f"usage:{tenant_id}"
        records = []
        
        if self.redis:
            try:
                import json
                raw_records = self.redis.lrange(key, 0, -1)
                for raw in raw_records:
                    data = json.loads(raw)
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])
                    records.append(UsageRecord(**data))
            except Exception as e:
                logger.warning(f"Redis usage fetch failed: {e}")
                records = self._memory_store.get(key, [])
        else:
            records = self._memory_store.get(key, [])
        
        # Filter
        total = 0
        for record in records:
            if resource_type and record.resource_type != resource_type:
                continue
            if start_date and record.timestamp < start_date:
                continue
            if end_date and record.timestamp > end_date:
                continue
            total += record.amount
        
        return total
    
    def get_summary(
        self,
        tenant_id: str,
        period_days: int = 30,
    ) -> UsageSummary:
        """
        Get complete usage summary for billing.
        
        Args:
            tenant_id: Tenant ID
            period_days: Number of days to summarize
            
        Returns:
            UsageSummary with all metrics
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=period_days)
        
        audio_hours = self.get_usage(tenant_id, "audio_hours", start_date, end_date)
        storage_mb = self.get_usage(tenant_id, "storage_mb", start_date, end_date)
        queries = self.get_usage(tenant_id, "queries", start_date, end_date)
        api_calls = self.get_usage(tenant_id, "api_calls", start_date, end_date)
        
        # Calculate estimated cost
        cost = (
            audio_hours * self.pricing["audio_hours"] +
            (storage_mb / 1024) * self.pricing["storage_gb"] +
            queries * self.pricing["queries"] +
            api_calls * self.pricing["api_calls"]
        )
        
        return UsageSummary(
            tenant_id=tenant_id,
            period_start=start_date,
            period_end=end_date,
            audio_hours=round(audio_hours, 2),
            storage_used_mb=round(storage_mb, 2),
            query_count=int(queries),
            api_call_count=int(api_calls),
            estimated_cost=round(cost, 2),
        )


# ===================================
# Global Instance
# ===================================

_usage_tracker: Optional[UsageTracker] = None


def init_usage_tracker(redis_client=None):
    """Initialize global usage tracker."""
    global _usage_tracker
    _usage_tracker = UsageTracker(redis_client)


def get_usage_tracker() -> UsageTracker:
    """Get global usage tracker."""
    global _usage_tracker
    if _usage_tracker is None:
        _usage_tracker = UsageTracker()
    return _usage_tracker


def get_usage_summary(tenant_id: str, period_days: int = 30) -> UsageSummary:
    """Get usage summary for tenant."""
    return get_usage_tracker().get_summary(tenant_id, period_days)
