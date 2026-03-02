"""RAG service — thin orchestration layer that routes to the active backend.

When ``BACKEND=local`` → uses ChromaDB (app.clients.local_chroma)
When ``BACKEND=nvidia`` → uses NVIDIA RAG HTTP (app.clients.nvidia_rag_http)
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Any

from app.config import settings
from app.schemas import (
    ChatQueryResponse,
    Chunk,
    Citation,
    SearchResponse,
)

logger = logging.getLogger(__name__)


def _backend() -> ModuleType:
    """Return the correct client module based on settings.backend."""
    if settings.backend == "local":
        from app.clients import local_chroma as client
    else:
        from app.clients import nvidia_rag_http as client
    return client


def _build_citations(chunks: list[dict[str, Any]]) -> list[Citation]:
    """Create a numbered citation list from normalised chunks."""
    return [
        Citation(
            i=idx + 1,
            source=c.get("source", ""),
            metadata=c.get("metadata", {}),
            score=c.get("score", 0.0),
            snippet=c.get("text", "")[:300],
        )
        for idx, c in enumerate(chunks)
    ]


async def search(query: str, collection: str = "", top_k: int = 5) -> SearchResponse:
    """Run a RAG search and return structured response."""
    client = _backend()
    col = collection or settings.rag_collection
    raw_chunks = await client.search(query=query, collection=col, top_k=top_k)
    chunks = [
        Chunk(
            text=c["text"],
            score=c["score"],
            source=c["source"],
            metadata=c["metadata"],
        )
        for c in raw_chunks
    ]
    citations = _build_citations(raw_chunks)
    return SearchResponse(query=query, chunks=chunks, citations=citations)


async def chat_query(query: str, collection: str = "", top_k: int = 5) -> ChatQueryResponse:
    """Ask a question through the RAG generate endpoint."""
    client = _backend()
    col = collection or settings.rag_collection
    result = await client.generate(query=query, collection=col, top_k=top_k)
    citations = _build_citations(result.get("chunks", []))
    return ChatQueryResponse(
        query=query,
        answer=result.get("answer", ""),
        citations=citations,
    )
