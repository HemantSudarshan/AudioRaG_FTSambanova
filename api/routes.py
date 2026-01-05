"""
FastAPI Routes

REST API endpoints for AudioRAG.
"""

import logging
from datetime import datetime
from typing import Optional, List
from dataclasses import asdict

from fastapi import FastAPI, APIRouter, Depends, HTTPException, UploadFile, File, Query, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from monitoring.health import get_health_status, get_liveness, get_readiness

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="AudioRAG API",
    description="REST API for Audio Analytics with RAG",
    version=settings.app_version,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# API Router
router = APIRouter(prefix="/api/v1")


# ===================================
# Schemas
# ===================================

class AudioUploadResponse(BaseModel):
    """Response for audio upload."""
    id: str
    name: str
    status: str
    created_at: str


class AudioDetailResponse(BaseModel):
    """Detailed audio information."""
    id: str
    name: str
    status: str
    duration_seconds: Optional[float] = None
    speakers_detected: Optional[int] = None
    transcript_segments: Optional[int] = None
    created_at: str


class QueryRequest(BaseModel):
    """Query request body."""
    audio_id: str = Field(..., description="ID of the audio to query")
    query: str = Field(..., min_length=1, description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")


class QueryResponse(BaseModel):
    """Query response."""
    answer: str
    sources: List[dict]
    query_time_ms: float


class AnalyticsResponse(BaseModel):
    """Analytics data response."""
    total_audio_hours: float
    total_queries: int
    average_response_time_ms: float
    active_users: int
    generated_at: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    code: str


# ===================================
# Dependencies
# ===================================

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """
    Verify API key from Authorization header.
    
    In production, this would validate against database.
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authorization header required"
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization format. Use 'Bearer <api_key>'"
        )
    
    api_key = authorization[7:]  # Remove "Bearer "
    
    # TODO: Verify against database
    if not api_key.startswith("ar_"):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key


# ===================================
# Health Endpoints
# ===================================

@app.get("/health", tags=["Health"])
async def health():
    """Kubernetes liveness probe."""
    return get_liveness()


@app.get("/ready", tags=["Health"])
async def ready():
    """Kubernetes readiness probe."""
    result = get_readiness()
    status_code = 200 if result["ready"] else 503
    return JSONResponse(content=result, status_code=status_code)


@app.get("/health/detailed", tags=["Health"])
async def health_detailed():
    """Detailed health status."""
    health = get_health_status(include_dependencies=True)
    return asdict(health)


# ===================================
# Audio Endpoints
# ===================================

@router.post(
    "/audio",
    response_model=AudioUploadResponse,
    tags=["Audio"],
    summary="Upload audio file",
)
async def upload_audio(
    file: UploadFile = File(...),
    name: Optional[str] = None,
    speakers: Optional[int] = Query(None, ge=1, le=10),
    api_key: str = Depends(verify_api_key),
):
    """
    Upload an audio file for processing.
    
    - **file**: Audio file (MP3, WAV, M4A)
    - **name**: Optional custom name
    - **speakers**: Expected number of speakers (optional)
    """
    # Validate file type
    allowed_types = ["audio/mpeg", "audio/wav", "audio/x-m4a", "audio/mp4"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: MP3, WAV, M4A"
        )
    
    # TODO: Process file asynchronously
    audio_id = f"audio_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    logger.info(f"Audio uploaded: {audio_id}")
    
    return AudioUploadResponse(
        id=audio_id,
        name=name or file.filename,
        status="processing",
        created_at=datetime.utcnow().isoformat(),
    )


@router.get(
    "/audio/{audio_id}",
    response_model=AudioDetailResponse,
    tags=["Audio"],
    summary="Get audio details",
)
async def get_audio(
    audio_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Get details of an uploaded audio file."""
    # TODO: Fetch from database
    
    # Mock response
    return AudioDetailResponse(
        id=audio_id,
        name="sample.mp3",
        status="completed",
        duration_seconds=180.5,
        speakers_detected=2,
        transcript_segments=45,
        created_at=datetime.utcnow().isoformat(),
    )


@router.get(
    "/audio/{audio_id}/transcript",
    tags=["Audio"],
    summary="Get transcript",
)
async def get_transcript(
    audio_id: str,
    format: str = Query("json", regex="^(json|txt|srt)$"),
    speaker: Optional[str] = None,
    api_key: str = Depends(verify_api_key),
):
    """
    Get transcript for an audio file.
    
    - **format**: Output format (json, txt, srt)
    - **speaker**: Filter by speaker
    """
    # TODO: Fetch from database
    
    segments = [
        {
            "speaker": "Speaker A",
            "text": "Hello, welcome to the interview.",
            "start_time": 0.0,
            "end_time": 3.5,
        }
    ]
    
    if format == "txt":
        text = "\n".join(f"[{s['speaker']}]: {s['text']}" for s in segments)
        return {"transcript": text}
    elif format == "srt":
        # TODO: Generate SRT format
        return {"transcript": "SRT format not implemented"}
    
    return {"segments": segments}


@router.delete(
    "/audio/{audio_id}",
    tags=["Audio"],
    summary="Delete audio",
)
async def delete_audio(
    audio_id: str,
    api_key: str = Depends(verify_api_key),
):
    """Delete an audio file and its data."""
    # TODO: Delete from database and vector store
    
    logger.info(f"Audio deleted: {audio_id}")
    
    return {"message": f"Audio {audio_id} deleted", "deleted_at": datetime.utcnow().isoformat()}


# ===================================
# Query Endpoints
# ===================================

@router.post(
    "/query",
    response_model=QueryResponse,
    tags=["Query"],
    summary="Query audio content",
)
async def query_audio(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key),
):
    """
    Query audio content using natural language.
    
    Returns an AI-generated answer with source segments.
    """
    import time
    start = time.time()
    
    # TODO: Execute RAG query
    
    query_time = (time.time() - start) * 1000
    
    return QueryResponse(
        answer="This is a sample response. Integrate with RAG engine for real responses.",
        sources=[
            {"text": "[00:30] Speaker A: Relevant segment...", "score": 0.85}
        ],
        query_time_ms=round(query_time, 2),
    )


# ===================================
# Analytics Endpoints
# ===================================

@router.get(
    "/analytics",
    response_model=AnalyticsResponse,
    tags=["Analytics"],
    summary="Get usage analytics",
)
async def get_analytics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    api_key: str = Depends(verify_api_key),
):
    """Get usage analytics for the authenticated user/tenant."""
    from analytics.metrics import get_metrics
    
    metrics = get_metrics()
    dashboard = metrics.get_dashboard_data()
    
    return AnalyticsResponse(
        total_audio_hours=dashboard.get("audio_duration_hours", {}).get("total", 0),
        total_queries=int(dashboard.get("queries_24h", {}).get("count", 0)),
        average_response_time_ms=dashboard.get("avg_query_latency_ms", {}).get("average", 0),
        active_users=dashboard.get("active_users", 0),
        generated_at=dashboard.get("generated_at", datetime.utcnow().isoformat()),
    )


# Register router
app.include_router(router)


# ===================================
# Error Handlers
# ===================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail),
            code=f"HTTP_{exc.status_code}",
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if settings.debug else None,
            code="INTERNAL_ERROR",
        ).dict(),
    )
