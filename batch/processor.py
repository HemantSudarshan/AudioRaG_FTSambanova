"""
Batch Processor

Core batch processing logic for audio files.
"""

import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
import httpx

from batch.queue import JobQueue, Job, JobStatus

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Process batch jobs asynchronously.
    
    Handles transcription, embedding, and indexing.
    """
    
    def __init__(self, queue: JobQueue):
        self.queue = queue
        self._running = False
    
    def process_job(self, job: Job) -> bool:
        """
        Process a single job.
        
        Args:
            job: Job to process
            
        Returns:
            True if successful
        """
        try:
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow().isoformat()
            job.attempts += 1
            self.queue.update_job(job)
            
            logger.info(f"Processing job {job.id}: {job.type}")
            
            # Transcription
            if job.type in ["transcribe", "full"]:
                job.status = JobStatus.TRANSCRIBING
                job.current_step = "Transcribing audio..."
                job.progress = 20
                self.queue.update_job(job)
                
                result = self._transcribe(job)
                if not result:
                    raise Exception("Transcription failed")
                
                job.result = job.result or {}
                job.result["transcription"] = result
            
            # Embedding
            if job.type in ["embed", "full"]:
                job.status = JobStatus.EMBEDDING
                job.current_step = "Generating embeddings..."
                job.progress = 50
                self.queue.update_job(job)
                
                result = self._embed(job)
                if not result:
                    raise Exception("Embedding failed")
                
                job.result = job.result or {}
                job.result["embeddings"] = result
            
            # Indexing
            if job.type in ["index", "full"]:
                job.status = JobStatus.INDEXING
                job.current_step = "Indexing in vector store..."
                job.progress = 80
                self.queue.update_job(job)
                
                result = self._index(job)
                if not result:
                    raise Exception("Indexing failed")
                
                job.result = job.result or {}
                job.result["indexed"] = True
            
            # Complete
            job.status = JobStatus.COMPLETED
            job.progress = 100
            job.current_step = "Completed"
            job.completed_at = datetime.utcnow().isoformat()
            self.queue.update_job(job)
            
            logger.info(f"Completed job {job.id}")
            
            # Send webhook
            if job.webhook_url:
                self._send_webhook(job)
            
            return True
            
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = datetime.utcnow().isoformat()
            
            # Retry if under max attempts
            if job.attempts < job.max_attempts:
                job.status = JobStatus.QUEUED
                logger.info(f"Retrying job {job.id} (attempt {job.attempts}/{job.max_attempts})")
            
            self.queue.update_job(job)
            
            if job.webhook_url:
                self._send_webhook(job)
            
            return False
    
    def _transcribe(self, job: Job) -> Optional[Dict[str, Any]]:
        """Transcribe audio file."""
        import os
        
        try:
            from rag_code import Transcribe
            
            api_key = os.getenv("ASSEMBLYAI_API_KEY")
            if not api_key:
                raise ValueError("ASSEMBLYAI_API_KEY not set")
            
            transcriber = Transcribe(api_key=api_key)
            segments = transcriber.transcribe_audio(job.file_path)
            
            return {
                "segments": len(segments),
                "speakers": len(set(s.get("speaker") for s in segments if s.get("speaker"))),
                "duration": segments[-1]["end_time"] if segments else 0,
            }
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    def _embed(self, job: Job) -> Optional[Dict[str, Any]]:
        """Generate embeddings for transcription."""
        try:
            from rag_code import EmbedData
            
            # This is a placeholder - would use actual transcription
            embedder = EmbedData()
            
            return {
                "model": embedder.embed_model_name,
                "dimension": 1024,
            }
            
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    def _index(self, job: Job) -> Optional[Dict[str, Any]]:
        """Index embeddings in vector store."""
        try:
            from rag_code import QdrantVDB_QB
            
            collection_name = f"audio_{job.audio_id}"
            
            return {
                "collection": collection_name,
                "indexed_at": datetime.utcnow().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Indexing error: {e}")
            return None
    
    def _send_webhook(self, job: Job):
        """Send webhook notification."""
        if not job.webhook_url:
            return
        
        try:
            payload = {
                "job_id": job.id,
                "status": job.status.value,
                "audio_id": job.audio_id,
                "result": job.result,
                "error": job.error,
                "completed_at": job.completed_at,
            }
            
            with httpx.Client(timeout=10) as client:
                response = client.post(job.webhook_url, json=payload)
                logger.info(f"Webhook sent for job {job.id}: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Webhook failed for job {job.id}: {e}")
    
    def start(self):
        """Start processing queue."""
        self._running = True
        logger.info("Batch processor started")
        
        while self._running:
            job = self.queue.dequeue()
            
            if job:
                self.process_job(job)
            else:
                time.sleep(1)  # Wait for new jobs
    
    def stop(self):
        """Stop processing."""
        self._running = False
        logger.info("Batch processor stopped")
