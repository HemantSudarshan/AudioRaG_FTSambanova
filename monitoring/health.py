"""
Health Check Endpoints

Kubernetes-ready liveness and readiness probes.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    last_check: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    version: str
    timestamp: str
    components: Dict[str, Dict[str, Any]]
    uptime_seconds: float


# Track application start time
_start_time = datetime.utcnow()


def check_qdrant_health(url: str = "http://localhost:6333") -> ComponentHealth:
    """
    Check Qdrant vector database health.
    
    Args:
        url: Qdrant server URL
        
    Returns:
        ComponentHealth status
    """
    import time
    
    try:
        from qdrant_client import QdrantClient
        
        start = time.time()
        client = QdrantClient(url=url, timeout=5)
        client.get_collections()
        latency = (time.time() - start) * 1000
        
        return ComponentHealth(
            name="qdrant",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message="Connected",
            last_check=datetime.utcnow().isoformat(),
        )
        
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        return ComponentHealth(
            name="qdrant",
            status=HealthStatus.UNHEALTHY,
            message=str(e),
            last_check=datetime.utcnow().isoformat(),
        )


def check_redis_health(url: str = "redis://localhost:6379") -> ComponentHealth:
    """
    Check Redis cache health.
    
    Args:
        url: Redis connection URL
        
    Returns:
        ComponentHealth status
    """
    import time
    
    try:
        import redis
        
        start = time.time()
        client = redis.from_url(url, socket_timeout=5)
        client.ping()
        latency = (time.time() - start) * 1000
        
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message="Connected",
            last_check=datetime.utcnow().isoformat(),
        )
        
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return ComponentHealth(
            name="redis",
            status=HealthStatus.DEGRADED,  # Redis is optional
            message=str(e),
            last_check=datetime.utcnow().isoformat(),
        )


def check_database_health(url: str = None) -> ComponentHealth:
    """
    Check database health.
    
    Args:
        url: Database connection URL
        
    Returns:
        ComponentHealth status
    """
    import time
    
    try:
        from sqlalchemy import create_engine, text
        
        if not url:
            url = "sqlite:///./audiorag.db"
        
        start = time.time()
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        latency = (time.time() - start) * 1000
        
        return ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message="Connected",
            last_check=datetime.utcnow().isoformat(),
        )
        
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=str(e),
            last_check=datetime.utcnow().isoformat(),
        )


def check_assemblyai_health() -> ComponentHealth:
    """
    Check AssemblyAI API health.
    
    Returns:
        ComponentHealth status
    """
    import os
    
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    
    if not api_key:
        return ComponentHealth(
            name="assemblyai",
            status=HealthStatus.UNHEALTHY,
            message="API key not configured",
            last_check=datetime.utcnow().isoformat(),
        )
    
    return ComponentHealth(
        name="assemblyai",
        status=HealthStatus.HEALTHY,
        message="API key configured",
        last_check=datetime.utcnow().isoformat(),
    )


def check_sambanova_health() -> ComponentHealth:
    """
    Check SambaNova API health.
    
    Returns:
        ComponentHealth status
    """
    import os
    
    api_key = os.getenv("SAMBANOVA_API_KEY")
    
    if not api_key:
        return ComponentHealth(
            name="sambanova",
            status=HealthStatus.DEGRADED,
            message="API key not configured (will use fallback)",
            last_check=datetime.utcnow().isoformat(),
        )
    
    return ComponentHealth(
        name="sambanova",
        status=HealthStatus.HEALTHY,
        message="API key configured",
        last_check=datetime.utcnow().isoformat(),
    )


def get_health_status(
    include_dependencies: bool = True,
    qdrant_url: str = "http://localhost:6333",
    redis_url: str = "redis://localhost:6379",
    database_url: str = None,
) -> SystemHealth:
    """
    Get comprehensive system health status.
    
    Args:
        include_dependencies: Check external dependencies
        qdrant_url: Qdrant server URL
        redis_url: Redis connection URL
        database_url: Database connection URL
        
    Returns:
        SystemHealth with all component statuses
    """
    from config import settings
    
    components = {}
    overall_status = HealthStatus.HEALTHY
    
    if include_dependencies:
        # Check Qdrant
        qdrant = check_qdrant_health(qdrant_url)
        components["qdrant"] = asdict(qdrant)
        
        # Check Redis
        redis = check_redis_health(redis_url)
        components["redis"] = asdict(redis)
        
        # Check Database
        db = check_database_health(database_url)
        components["database"] = asdict(db)
        
        # Check APIs
        assemblyai = check_assemblyai_health()
        components["assemblyai"] = asdict(assemblyai)
        
        sambanova = check_sambanova_health()
        components["sambanova"] = asdict(sambanova)
        
        # Determine overall status
        statuses = [c.status for c in [qdrant, redis, db, assemblyai, sambanova]]
        
        if HealthStatus.UNHEALTHY in statuses:
            # Critical services down
            if qdrant.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            elif assemblyai.status == HealthStatus.UNHEALTHY:
                overall_status = HealthStatus.UNHEALTHY
            else:
                overall_status = HealthStatus.DEGRADED
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
    
    uptime = (datetime.utcnow() - _start_time).total_seconds()
    
    return SystemHealth(
        status=overall_status,
        version=settings.app_version,
        timestamp=datetime.utcnow().isoformat(),
        components=components,
        uptime_seconds=round(uptime, 2),
    )


def get_liveness() -> Dict[str, str]:
    """
    Kubernetes liveness probe.
    
    Simple check - application is running.
    
    Returns:
        Status dict
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_readiness() -> Dict[str, Any]:
    """
    Kubernetes readiness probe.
    
    Full check - application can serve traffic.
    
    Returns:
        Status dict with readiness state
    """
    health = get_health_status(include_dependencies=True)
    
    is_ready = health.status != HealthStatus.UNHEALTHY
    
    return {
        "ready": is_ready,
        "status": health.status.value,
        "timestamp": datetime.utcnow().isoformat(),
    }
