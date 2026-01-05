"""
AudioRAG Configuration Management

Centralized configuration using Pydantic Settings with environment variable support.
Supports development, staging, and production environments.
"""

import os
from typing import Optional, Literal
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Environment
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment"
    )
    debug: bool = Field(default=True, description="Enable debug mode")
    
    # Application
    app_name: str = Field(default="AudioRAG", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    secret_key: str = Field(
        default="change-me-in-production",
        description="Secret key for JWT signing"
    )
    
    # AssemblyAI
    assemblyai_api_key: str = Field(
        ..., 
        description="AssemblyAI API key for transcription"
    )
    
    # SambaNova
    sambanova_api_key: Optional[str] = Field(
        default=None,
        description="SambaNova API key for LLM"
    )
    llm_model: str = Field(
        default="Meta-Llama-3.1-405B-Instruct",
        description="LLM model name"
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature"
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=1,
        le=8192,
        description="Maximum tokens in LLM response"
    )
    
    # OpenAI (fallback)
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (fallback)"
    )
    
    # Qdrant Vector Database
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key (for cloud)"
    )
    collection_name: str = Field(
        default="chat_with_audios",
        description="Default Qdrant collection name"
    )
    
    # Embedding Model
    embed_model_name: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="HuggingFace embedding model"
    )
    embed_batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Embedding batch size"
    )
    vector_dim: int = Field(
        default=1024,
        description="Embedding vector dimension"
    )
    
    # Redis Cache
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL"
    )
    cache_ttl: int = Field(
        default=3600,
        description="Default cache TTL in seconds"
    )
    
    # PostgreSQL Database
    database_url: str = Field(
        default="sqlite:///./audiorag.db",
        description="Database connection URL"
    )
    
    # Audio Processing
    max_file_size_mb: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum upload file size in MB"
    )
    supported_formats: list[str] = Field(
        default=["mp3", "wav", "m4a"],
        description="Supported audio formats"
    )
    
    # Retrieval Settings
    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to retrieve"
    )
    score_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(
        default=100,
        description="Requests per hour limit"
    )
    rate_limit_window: int = Field(
        default=3600,
        description="Rate limit window in seconds"
    )
    
    # Monitoring
    sentry_dsn: Optional[str] = Field(
        default=None,
        description="Sentry DSN for error tracking"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: str = Field(
        default="rag_audio.log",
        description="Log file path"
    )
    
    # JWT Settings
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm"
    )
    jwt_expiry_minutes: int = Field(
        default=60,
        description="JWT token expiry in minutes"
    )
    jwt_refresh_expiry_days: int = Field(
        default=7,
        description="JWT refresh token expiry in days"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Configuration for different environments
class DevelopmentSettings(Settings):
    """Development environment settings."""
    debug: bool = True
    log_level: str = "DEBUG"
    

class ProductionSettings(Settings):
    """Production environment settings."""
    debug: bool = False
    log_level: str = "WARNING"


def get_config() -> Settings:
    """Get configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return ProductionSettings()
    elif env == "staging":
        return Settings(environment="staging", debug=False)
    else:
        return DevelopmentSettings()


# Export settings instance
settings = get_settings()


# Legacy CONFIG dict for backward compatibility
CONFIG = {
    "collection_name": settings.collection_name,
    "embed_model_name": settings.embed_model_name,
    "llm_name": settings.llm_model,
    "vector_dim": settings.vector_dim,
    "embed_batch_size": settings.embed_batch_size,
    "qdrant_batch_size": 256,
    "max_file_size_mb": settings.max_file_size_mb,
    "qdrant_url": settings.qdrant_url,
    "supported_formats": settings.supported_formats,
    "min_speaker_confidence": 0.5,
    "min_audio_duration_for_multi_speaker": 30,
}
