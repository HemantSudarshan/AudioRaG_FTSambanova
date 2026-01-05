"""
AudioRAG API Module

REST API layer with FastAPI.
"""

from api.routes import router, app
from api.middleware import setup_middleware

__all__ = [
    "router",
    "app",
    "setup_middleware",
]
