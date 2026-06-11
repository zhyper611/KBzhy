"""Shared RAG engine instance for API modules."""

from __future__ import annotations

from KBzhy.app.core.rag_engine import RAGEngine


_engine: RAGEngine | None = None


def get_rag_engine() -> RAGEngine:
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine
