"""
Database Connection Management

SQLAlchemy engine and session handling.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

from config import settings

logger = logging.getLogger(__name__)

# Base class for models
Base = declarative_base()

# Engine instance (lazy initialization)
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create SQLAlchemy engine."""
    global _engine
    
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
        logger.info(f"Database engine created: {settings.database_url}")
    
    return _engine


def get_session_factory():
    """Get session factory."""
    global _SessionLocal
    
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=get_engine(),
        )
    
    return _SessionLocal


def get_session() -> Generator[Session, None, None]:
    """
    Get database session.
    
    Usage:
        for session in get_session():
            session.query(...)
    
    Or as dependency:
        def endpoint(db: Session = Depends(get_session)):
            ...
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    
    try:
        yield session
    finally:
        session.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Usage:
        with db_session() as session:
            session.query(...)
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database():
    """
    Initialize database and create all tables.
    
    Call this on application startup.
    """
    from auth.models import Base as AuthBase
    from audit.logger import Base as AuditBase
    
    engine = get_engine()
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    AuthBase.metadata.create_all(bind=engine)
    AuditBase.metadata.create_all(bind=engine)
    
    logger.info("Database tables created")


def drop_database():
    """Drop all database tables. Use with caution!"""
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    logger.warning("Database tables dropped")
