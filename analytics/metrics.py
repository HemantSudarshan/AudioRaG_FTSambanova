"""
Metrics Collection

Track usage metrics for analytics and billing.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics tracked."""
    AUDIO_UPLOADED = "audio_uploaded"
    AUDIO_DURATION = "audio_duration"
    TRANSCRIPTION_TIME = "transcription_time"
    QUERY_COUNT = "query_count"
    QUERY_LATENCY = "query_latency"
    LLM_TOKENS = "llm_tokens"
    EMBEDDING_COUNT = "embedding_count"
    ACTIVE_USERS = "active_users"
    API_CALLS = "api_calls"


@dataclass
class MetricEvent:
    """A single metric event."""
    metric_type: str
    value: float
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""
    metric_type: str
    total: float
    count: int
    average: float
    min_value: float
    max_value: float
    period_start: datetime
    period_end: datetime


class MetricsCollector:
    """
    Collect and aggregate usage metrics.
    
    In production, this would integrate with Redis or a time-series database.
    For simplicity, uses in-memory storage with optional Redis backing.
    """
    
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self._memory_store: Dict[str, List[MetricEvent]] = defaultdict(list)
        self._active_users: Dict[str, datetime] = {}
        logger.info("Metrics collector initialized")
    
    def record(
        self,
        metric_type: MetricType,
        value: float,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Record a metric event.
        
        Args:
            metric_type: Type of metric
            value: Metric value
            user_id: Associated user
            tenant_id: Associated tenant
            metadata: Additional data
        """
        event = MetricEvent(
            metric_type=metric_type.value,
            value=value,
            user_id=user_id,
            tenant_id=tenant_id,
            metadata=metadata,
        )
        
        # Store in memory
        key = f"{metric_type.value}:{tenant_id or 'global'}"
        self._memory_store[key].append(event)
        
        # Track active users
        if user_id:
            self._active_users[user_id] = datetime.utcnow()
        
        # If Redis available, also store there
        if self.redis:
            try:
                import json
                redis_key = f"metrics:{key}:{datetime.utcnow().strftime('%Y%m%d%H')}"
                self.redis.lpush(redis_key, json.dumps(asdict(event), default=str))
                self.redis.expire(redis_key, 86400 * 30)  # 30 days retention
            except Exception as e:
                logger.warning(f"Failed to store metric in Redis: {e}")
        
        logger.debug(f"Recorded metric: {metric_type.value}={value}")
    
    def get_summary(
        self,
        metric_type: MetricType,
        tenant_id: Optional[str] = None,
        period_hours: int = 24,
    ) -> MetricSummary:
        """
        Get summary statistics for a metric.
        
        Args:
            metric_type: Type of metric
            tenant_id: Filter by tenant
            period_hours: Time period to summarize
            
        Returns:
            MetricSummary with aggregated stats
        """
        key = f"{metric_type.value}:{tenant_id or 'global'}"
        events = self._memory_store.get(key, [])
        
        # Filter by time period
        cutoff = datetime.utcnow() - timedelta(hours=period_hours)
        events = [e for e in events if e.timestamp >= cutoff]
        
        if not events:
            return MetricSummary(
                metric_type=metric_type.value,
                total=0,
                count=0,
                average=0,
                min_value=0,
                max_value=0,
                period_start=cutoff,
                period_end=datetime.utcnow(),
            )
        
        values = [e.value for e in events]
        
        return MetricSummary(
            metric_type=metric_type.value,
            total=sum(values),
            count=len(values),
            average=sum(values) / len(values),
            min_value=min(values),
            max_value=max(values),
            period_start=cutoff,
            period_end=datetime.utcnow(),
        )
    
    def get_active_users(
        self,
        tenant_id: Optional[str] = None,
        period_minutes: int = 30,
    ) -> int:
        """
        Get count of active users.
        
        Args:
            tenant_id: Filter by tenant
            period_minutes: Activity window
            
        Returns:
            Number of active users
        """
        cutoff = datetime.utcnow() - timedelta(minutes=period_minutes)
        active = sum(1 for ts in self._active_users.values() if ts >= cutoff)
        return active
    
    def get_dashboard_data(
        self,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get all data needed for analytics dashboard.
        
        Args:
            tenant_id: Filter by tenant
            
        Returns:
            Dashboard data dict
        """
        return {
            "audio_uploads_24h": asdict(self.get_summary(MetricType.AUDIO_UPLOADED, tenant_id)),
            "queries_24h": asdict(self.get_summary(MetricType.QUERY_COUNT, tenant_id)),
            "audio_duration_hours": asdict(self.get_summary(MetricType.AUDIO_DURATION, tenant_id)),
            "avg_query_latency_ms": asdict(self.get_summary(MetricType.QUERY_LATENCY, tenant_id)),
            "active_users": self.get_active_users(tenant_id),
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    def cleanup_old_data(self, max_age_hours: int = 168):
        """
        Clean up old metric data from memory.
        
        Args:
            max_age_hours: Maximum age of data to keep (default 7 days)
        """
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        for key in self._memory_store:
            self._memory_store[key] = [
                e for e in self._memory_store[key]
                if e.timestamp >= cutoff
            ]
        
        # Clean active users
        self._active_users = {
            uid: ts for uid, ts in self._active_users.items()
            if ts >= cutoff
        }


# ===================================
# Global Metrics Instance
# ===================================

_metrics: Optional[MetricsCollector] = None


def init_metrics(redis_client=None):
    """Initialize global metrics collector."""
    global _metrics
    _metrics = MetricsCollector(redis_client)


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


# ===================================
# Convenience Functions
# ===================================

def track_audio_upload(
    file_size_mb: float,
    duration_seconds: float,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
):
    """Track an audio upload event."""
    metrics = get_metrics()
    metrics.record(
        MetricType.AUDIO_UPLOADED, 1,
        user_id=user_id, tenant_id=tenant_id,
        metadata={"file_size_mb": file_size_mb}
    )
    metrics.record(
        MetricType.AUDIO_DURATION, duration_seconds / 3600,  # Convert to hours
        user_id=user_id, tenant_id=tenant_id,
    )


def track_transcription(
    processing_time_seconds: float,
    segments_count: int,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
):
    """Track a transcription event."""
    metrics = get_metrics()
    metrics.record(
        MetricType.TRANSCRIPTION_TIME, processing_time_seconds,
        user_id=user_id, tenant_id=tenant_id,
        metadata={"segments": segments_count}
    )


def track_query(
    latency_ms: float,
    tokens_used: int = 0,
    user_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
):
    """Track a query event."""
    metrics = get_metrics()
    metrics.record(
        MetricType.QUERY_COUNT, 1,
        user_id=user_id, tenant_id=tenant_id,
    )
    metrics.record(
        MetricType.QUERY_LATENCY, latency_ms,
        user_id=user_id, tenant_id=tenant_id,
    )
    if tokens_used:
        metrics.record(
            MetricType.LLM_TOKENS, tokens_used,
            user_id=user_id, tenant_id=tenant_id,
        )
