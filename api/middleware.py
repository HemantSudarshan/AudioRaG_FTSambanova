"""
API Middleware

Rate limiting, CORS, and request logging.
"""

import logging
import time
from typing import Callable
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiting middleware.
    
    In production, use Redis for distributed rate limiting.
    """
    
    def __init__(
        self,
        app,
        requests_per_hour: int = 100,
        requests_per_minute: int = 20,
    ):
        super().__init__(app)
        self.requests_per_hour = requests_per_hour
        self.requests_per_minute = requests_per_minute
        self._request_counts: dict = defaultdict(list)
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier (API key or IP)."""
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            return f"key:{auth[7:20]}"  # Use first 13 chars of key
        
        # Fall back to IP
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        return f"ip:{request.client.host}"
    
    def _is_rate_limited(self, client_id: str) -> tuple:
        """Check if client is rate limited."""
        now = datetime.utcnow()
        
        # Clean old entries
        hour_ago = now - timedelta(hours=1)
        minute_ago = now - timedelta(minutes=1)
        
        self._request_counts[client_id] = [
            ts for ts in self._request_counts[client_id]
            if ts > hour_ago
        ]
        
        requests = self._request_counts[client_id]
        
        # Check minute limit
        minute_requests = sum(1 for ts in requests if ts > minute_ago)
        if minute_requests >= self.requests_per_minute:
            return True, "minute", self.requests_per_minute
        
        # Check hour limit
        if len(requests) >= self.requests_per_hour:
            return True, "hour", self.requests_per_hour
        
        return False, None, None
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip rate limiting for health endpoints
        if request.url.path in ["/health", "/ready"]:
            return await call_next(request)
        
        client_id = self._get_client_id(request)
        is_limited, period, limit = self._is_rate_limited(client_id)
        
        if is_limited:
            logger.warning(f"Rate limited: {client_id}")
            return Response(
                content=f'{{"error": "Rate limit exceeded", "limit": {limit}, "period": "{period}"}}',
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60" if period == "minute" else "3600",
                },
            )
        
        # Record request
        self._request_counts[client_id].append(datetime.utcnow())
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        remaining = self.requests_per_hour - len(self._request_counts[client_id])
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_hour)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests with timing."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start) * 1000
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} "
            f"duration={duration_ms:.2f}ms"
        )
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        return response


def setup_middleware(app: FastAPI):
    """Configure all middleware for the FastAPI app."""
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_hour=settings.rate_limit_requests,
        requests_per_minute=20,
    )
    
    # Request logging
    app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("API middleware configured")
