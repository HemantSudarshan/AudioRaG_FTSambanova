"""
Database Models

SQLAlchemy models for audio files, transcripts, and queries.
"""

from datetime import datetime
from typing import Optional, List

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column

from database.connection import Base


class AudioFile(Base):
    """Audio file metadata and status."""
    __tablename__ = "audio_files"
    
    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # User/Tenant
    user_id: Mapped[Optional[str]] = mapped_column(String(50), index=True, nullable=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(50), index=True, nullable=True)
    
    # File info
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Processing status
    status: Mapped[str] = mapped_column(String(50), default="pending", index=True)
    # pending, processing, transcribing, embedding, completed, failed
    
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Qdrant collection
    collection_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    transcripts = relationship("Transcript", back_populates="audio_file", cascade="all, delete-orphan")
    queries = relationship("Query", back_populates="audio_file", cascade="all, delete-orphan")


class Transcript(Base):
    """Transcribed segments from an audio file."""
    __tablename__ = "transcripts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_id: Mapped[str] = mapped_column(String(50), ForeignKey("audio_files.id"), nullable=False)
    
    # Segment info
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    speaker: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Timestamps
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Confidence
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Metadata
    metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Parent
    audio_file = relationship("AudioFile", back_populates="transcripts")


class Query(Base):
    """User queries against audio files."""
    __tablename__ = "queries"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_id: Mapped[str] = mapped_column(String(50), ForeignKey("audio_files.id"), nullable=False)
    
    # User info
    user_id: Mapped[Optional[str]] = mapped_column(String(50), index=True, nullable=True)
    
    # Query
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Response
    response_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Performance
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Parent
    audio_file = relationship("AudioFile", back_populates="queries")
    
    # Results
    results = relationship("QueryResult", back_populates="query", cascade="all, delete-orphan")


class QueryResult(Base):
    """Retrieved segments for a query."""
    __tablename__ = "query_results"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_id: Mapped[int] = mapped_column(Integer, ForeignKey("queries.id"), nullable=False)
    
    # Retrieved segment
    segment_text: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Parent
    query = relationship("Query", back_populates="results")
