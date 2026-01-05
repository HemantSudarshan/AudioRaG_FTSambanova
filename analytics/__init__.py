"""
AudioRAG Analytics Module

Usage metrics, dashboards, and reporting.
"""

from analytics.metrics import (
    MetricsCollector,
    track_audio_upload,
    track_query,
    track_transcription,
)

__all__ = [
    "MetricsCollector",
    "track_audio_upload",
    "track_query",
    "track_transcription",
]
