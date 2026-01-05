"""
Tenant Models

SQLAlchemy models for multi-tenant architecture.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field

Base = declarative_base()


class TenantPlan(str, Enum):
    """Subscription plans."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class Tenant(Base):
    """Organization/Tenant model."""
    __tablename__ = "tenants"
    
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    
    # Contact
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Plan
    plan: Mapped[str] = mapped_column(String(50), default=TenantPlan.FREE.value)
    
    # Limits
    max_users: Mapped[int] = mapped_column(Integer, default=5)
    max_audio_hours: Mapped[int] = mapped_column(Integer, default=10)  # Per month
    max_storage_gb: Mapped[int] = mapped_column(Integer, default=5)
    
    # Usage
    current_audio_hours: Mapped[float] = mapped_column(Integer, default=0)
    current_storage_used_mb: Mapped[float] = mapped_column(Integer, default=0)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_trial: Mapped[bool] = mapped_column(Boolean, default=True)
    trial_ends_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Qdrant
    collection_prefix: Mapped[str] = mapped_column(String(50), nullable=True)
    
    # Settings
    settings: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class TenantSettings(BaseModel):
    """Tenant configuration settings."""
    
    # Feature flags
    enable_speaker_diarization: bool = True
    enable_sentiment_analysis: bool = False
    enable_custom_vocabulary: bool = False
    enable_domain_models: bool = False
    
    # Processing
    default_language: str = "en"
    transcription_model: str = "default"
    llm_model: str = "Meta-Llama-3.1-405B-Instruct"
    
    # Retention
    audio_retention_days: int = 90
    transcript_retention_days: int = 365
    
    # Webhooks
    webhook_enabled: bool = False
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    
    # Compliance
    hipaa_mode: bool = False
    gdpr_mode: bool = False
    data_region: str = "us"


# ===================================
# Pydantic Schemas
# ===================================

class TenantCreate(BaseModel):
    """Schema for creating a tenant."""
    name: str = Field(..., min_length=2, max_length=255)
    email: str
    plan: TenantPlan = TenantPlan.FREE


class TenantResponse(BaseModel):
    """Schema for tenant response."""
    id: str
    name: str
    slug: str
    email: str
    plan: str
    is_active: bool
    is_trial: bool
    max_audio_hours: int
    current_audio_hours: float
    created_at: datetime
    
    class Config:
        from_attributes = True


# ===================================
# Plan Limits
# ===================================

PLAN_LIMITS = {
    TenantPlan.FREE: {
        "max_users": 2,
        "max_audio_hours": 5,
        "max_storage_gb": 1,
        "features": ["basic_transcription", "basic_rag"],
    },
    TenantPlan.STARTER: {
        "max_users": 10,
        "max_audio_hours": 50,
        "max_storage_gb": 25,
        "features": ["basic_transcription", "basic_rag", "speaker_diarization", "analytics"],
    },
    TenantPlan.PROFESSIONAL: {
        "max_users": 50,
        "max_audio_hours": 500,
        "max_storage_gb": 100,
        "features": ["all_transcription", "advanced_rag", "speaker_diarization", 
                     "analytics", "api_access", "custom_vocabulary"],
    },
    TenantPlan.ENTERPRISE: {
        "max_users": -1,  # Unlimited
        "max_audio_hours": -1,  # Unlimited
        "max_storage_gb": -1,  # Unlimited
        "features": ["all_transcription", "advanced_rag", "speaker_diarization",
                     "analytics", "api_access", "custom_vocabulary", "domain_models",
                     "sso", "audit_logs", "hipaa", "on_prem"],
    },
}
