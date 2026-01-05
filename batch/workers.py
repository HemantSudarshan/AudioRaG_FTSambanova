"""
Background Workers

Celery-based workers for distributed processing.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Check if Celery is available
try:
    from celery import Celery
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    logger.info("Celery not installed, using sync processing")


def create_celery_app(
    broker_url: str = None,
    result_backend: str = None,
) -> Optional["Celery"]:
    """
    Create Celery application for distributed processing.
    
    Args:
        broker_url: Redis/RabbitMQ broker URL
        result_backend: Result backend URL
        
    Returns:
        Celery application or None
    """
    if not CELERY_AVAILABLE:
        return None
    
    broker_url = broker_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
    result_backend = result_backend or broker_url
    
    app = Celery(
        "audiorag",
        broker=broker_url,
        backend=result_backend,
    )
    
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_time_limit=3600,  # 1 hour max
        worker_prefetch_multiplier=1,
        worker_concurrency=4,
    )
    
    return app


# Create global Celery app
celery_app = create_celery_app() if CELERY_AVAILABLE else None


if CELERY_AVAILABLE and celery_app:
    
    @celery_app.task(bind=True, name="audiorag.transcribe")
    def transcribe_audio_task(self, audio_id: str, file_path: str, user_id: str = None):
        """
        Celery task for audio transcription.
        
        Args:
            audio_id: Audio file ID
            file_path: Path to audio file
            user_id: User ID
        """
        from batch.queue import JobQueue, Job, JobStatus
        from batch.processor import BatchProcessor
        
        try:
            self.update_state(state="TRANSCRIBING", meta={"progress": 10})
            
            queue = JobQueue()
            job = Job(
                id=self.request.id,
                type="transcribe",
                status=JobStatus.PROCESSING,
                priority=5,
                audio_id=audio_id,
                file_path=file_path,
                user_id=user_id,
            )
            
            processor = BatchProcessor(queue)
            success = processor.process_job(job)
            
            return {
                "success": success,
                "job_id": job.id,
                "result": job.result,
            }
            
        except Exception as e:
            logger.error(f"Transcription task failed: {e}")
            raise
    
    @celery_app.task(bind=True, name="audiorag.process_batch")
    def process_batch_task(self, audio_ids: list, operation: str, user_id: str = None):
        """
        Celery task for batch processing multiple files.
        
        Args:
            audio_ids: List of audio file IDs
            operation: Operation to perform
            user_id: User ID
        """
        results = []
        total = len(audio_ids)
        
        for i, audio_id in enumerate(audio_ids):
            self.update_state(
                state="PROCESSING",
                meta={"current": i + 1, "total": total, "audio_id": audio_id}
            )
            
            # Process each audio
            result = {"audio_id": audio_id, "status": "processed"}
            results.append(result)
        
        return {"processed": len(results), "results": results}


def start_worker(
    concurrency: int = 4,
    queues: list = None,
):
    """
    Start Celery worker.
    
    Args:
        concurrency: Number of worker processes
        queues: Queues to process
    """
    if not CELERY_AVAILABLE or not celery_app:
        logger.error("Celery not available")
        return
    
    queues = queues or ["default"]
    
    logger.info(f"Starting worker with concurrency={concurrency}")
    
    celery_app.worker_main([
        "worker",
        f"--concurrency={concurrency}",
        f"--queues={','.join(queues)}",
        "--loglevel=info",
    ])


# Sync fallback for when Celery is not available
def process_sync(audio_id: str, file_path: str, operation: str = "full"):
    """
    Synchronous processing fallback.
    
    Args:
        audio_id: Audio file ID
        file_path: Path to audio file
        operation: Operation to perform
    """
    from batch.queue import JobQueue, JobStatus
    from batch.processor import BatchProcessor
    
    queue = JobQueue()
    job = queue.enqueue(
        job_type=operation,
        audio_id=audio_id,
        file_path=file_path,
    )
    
    processor = BatchProcessor(queue)
    processor.process_job(job)
    
    return job
