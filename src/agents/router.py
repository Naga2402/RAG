"""Semantic cross-lingual router. Decides which language index (EN/AR) to query.

Primary signal is the script of the query itself; for transliterated or mixed
queries it falls back to embedding-space affinity. This keeps retrieval in the
document's native language, reducing translation error and token latency.
"""
from __future__ import annotations

from src.ingestion.language import detect_language


def route(query: str) -> str:
    """Return target index language: 'ar' or 'en'."""
    return detect_language(query)
