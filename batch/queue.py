"""
Job Queue Management

Redis-backed job queue for async processing.
"""

import logging
import json
import uuid
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job processing status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    TRANSCRIBING = "transcribing"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    URGENT = 20


@dataclass
class Job:
    """Batch processing job."""
    id: str
    type: str  # "transcribe", "embed", "summarize"
    status: JobStatus
    priority: int
    
    # Input
    audio_id: str
    file_path: Optional[str] = None
    
    # User/Tenant
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    
    # Processing
    progress: int = 0  # 0-100
    current_step: Optional[str] = None
    
    # Output
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # Timing
    created_at: str = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    
    # Retry
    attempts: int = 0
    max_attempts: int = 3
    
    # Webhook
    webhook_url: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()


class JobQueue:
    """
    Redis-backed job queue with priority support.
    
    Falls back to in-memory queue when Redis unavailable.
    """
    
    def __init__(self, redis_client=None, queue_name: str = "audiorag:jobs"):
        self.redis = redis_client
        self.queue_name = queue_name
        self._memory_queue: List[Job] = []
        self._jobs: Dict[str, Job] = {}
        logger.info(f"Job queue initialized: {queue_name}")
    
    def _use_redis(self) -> bool:
        """Check if Redis is available."""
        if not self.redis:
            return False
        try:
            self.redis.ping()
            return True
        except Exception:
            return False
    
    def enqueue(
        self,
        job_type: str,
        audio_id: str,
        file_path: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        priority: int = JobPriority.NORMAL,
        webhook_url: Optional[str] = None,
    ) -> Job:
        """
        Add a job to the queue.
        
        Args:
            job_type: Type of job
            audio_id: Audio file ID
            file_path: Path to audio file
            user_id: User ID
            tenant_id: Tenant ID
            priority: Job priority
            webhook_url: URL for completion webhook
            
        Returns:
            Created job
        """
        job = Job(
            id=f"job_{uuid.uuid4().hex[:12]}",
            type=job_type,
            status=JobStatus.QUEUED,
            priority=priority,
            audio_id=audio_id,
            file_path=file_path,
            user_id=user_id,
            tenant_id=tenant_id,
            webhook_url=webhook_url,
        )
        
        if self._use_redis():
            # Store job details
            self.redis.hset(
                f"{self.queue_name}:job:{job.id}",
                mapping={"data": json.dumps(asdict(job))}
            )
            # Add to priority queue
            self.redis.zadd(
                f"{self.queue_name}:pending",
                {job.id: priority}
            )
        else:
            self._jobs[job.id] = job
            self._memory_queue.append(job)
            self._memory_queue.sort(key=lambda j: j.priority, reverse=True)
        
        logger.info(f"Enqueued job {job.id}: {job_type} for audio {audio_id}")
        return job
    
    def dequeue(self) -> Optional[Job]:
        """
        Get next job from queue.
        
        Returns:
            Next job or None if queue empty
        """
        if self._use_redis():
            # Get highest priority job
            result = self.redis.zpopmax(f"{self.queue_name}:pending")
            if not result:
                return None
            
            job_id = result[0][0].decode() if isinstance(result[0][0], bytes) else result[0][0]
            job_data = self.redis.hget(f"{self.queue_name}:job:{job_id}", "data")
            
            if job_data:
                data = json.loads(job_data)
                return Job(**data)
            return None
        else:
            if not self._memory_queue:
                return None
            return self._memory_queue.pop(0)
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        if self._use_redis():
            job_data = self.redis.hget(f"{self.queue_name}:job:{job_id}", "data")
            if job_data:
                data = json.loads(job_data)
                return Job(**data)
            return None
        else:
            return self._jobs.get(job_id)
    
    def update_job(self, job: Job):
        """Update job status."""
        if self._use_redis():
            self.redis.hset(
                f"{self.queue_name}:job:{job.id}",
                mapping={"data": json.dumps(asdict(job))}
            )
        else:
            self._jobs[job.id] = job
        
        logger.debug(f"Updated job {job.id}: {job.status}")
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        if self._use_redis():
            pending = self.redis.zcard(f"{self.queue_name}:pending")
            return {
                "pending": pending,
                "queue_name": self.queue_name,
            }
        else:
            return {
                "pending": len(self._memory_queue),
                "total_jobs": len(self._jobs),
            }
    
    def get_user_jobs(
        self,
        user_id: str,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> List[Job]:
        """Get jobs for a user."""
        jobs = []
        
        if self._use_redis():
            # Scan for user's jobs
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(
                    cursor,
                    match=f"{self.queue_name}:job:*",
                    count=100
                )
                
                for key in keys:
                    job_data = self.redis.hget(key, "data")
                    if job_data:
                        data = json.loads(job_data)
                        if data.get("user_id") == user_id:
                            job = Job(**data)
                            if status is None or job.status == status:
                                jobs.append(job)
                
                if cursor == 0 or len(jobs) >= limit:
                    break
        else:
            for job in self._jobs.values():
                if job.user_id == user_id:
                    if status is None or job.status == status:
                        jobs.append(job)
                        if len(jobs) >= limit:
                            break
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]
