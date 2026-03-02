"""HTTP client wrapper for NVIDIA RAG Blueprint (rag-server + ingestor-server).

This module centralises *all* HTTP interactions with the blueprint so the rest
of the codebase never constructs raw URLs.  Three path variables control the
exact endpoint routes — change them here (or via env) and everything adapts.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Any, BinaryIO

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ── Configurable endpoint paths ────────────────────────────────────
# Override via .env: RAG_SEARCH_PATH, RAG_GENERATE_PATH, INGEST_DOCUMENTS_PATH

SEARCH_PATH: str = settings.rag_search_path       # default "/search"
GENERATE_PATH: str = settings.rag_generate_path    # default "/generate"
INGEST_PATH: str = settings.ingest_documents_path  # default "/documents"

_TIMEOUT = httpx.Timeout(timeout=float(settings.request_timeout))


# ── Helpers ────────────────────────────────────────────────────────


def _url(base: str, path: str) -> str:
    """Join base URL and path, avoiding double slashes."""
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _normalize_chunks(raw: Any) -> list[dict[str, Any]]:
    """Extract a list of chunk dicts from various RAG response shapes.

    The blueprint might return ``chunks``, ``results``, or ``documents``.
    Each item might use ``text`` or ``content`` for the main field.
    """
    if isinstance(raw, dict):
        items: list[Any] = (
            raw.get("chunks")
            or raw.get("results")
            or raw.get("documents")
            or raw.get("data")
            or []
        )
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    normalised: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("content") or item.get("page_content") or ""
        score = float(item.get("score", item.get("relevance_score", 0.0)))
        source = (
            item.get("source")
            or item.get("metadata", {}).get("source")
            or item.get("metadata", {}).get("path")
            or ""
        )
        metadata = item.get("metadata") or {}
        normalised.append(
            {"text": str(text), "score": score, "source": str(source), "metadata": metadata}
        )
    return normalised


def _normalize_answer(raw: Any) -> str:
    """Extract an answer string from a generate response."""
    if isinstance(raw, dict):
        return str(
            raw.get("answer")
            or raw.get("response")
            or raw.get("content")
            or raw.get("text")
            or raw.get("choices", [{}])[0].get("message", {}).get("content", "")
            or ""
        )
    return str(raw)


# ── Public API ─────────────────────────────────────────────────────


async def search(
    query: str,
    collection: str = "",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Call the RAG server search endpoint and return normalised chunks."""
    url = _url(settings.rag_url, SEARCH_PATH)
    payload: dict[str, Any] = {"query": query, "top_k": top_k}
    if collection:
        payload["collection"] = collection

    logger.info("RAG search → %s  query=%r  top_k=%d", url, query[:80], top_k)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    chunks = _normalize_chunks(data)
    logger.info("RAG search returned %d chunks", len(chunks))
    return chunks


async def generate(
    query: str,
    collection: str = "",
    top_k: int = 5,
) -> dict[str, Any]:
    """Call the RAG server generate endpoint (retrieval + generation).

    Returns ``{"answer": str, "chunks": [...]}``.
    """
    url = _url(settings.rag_url, GENERATE_PATH)
    payload: dict[str, Any] = {"query": query, "top_k": top_k}
    if collection:
        payload["collection"] = collection

    logger.info("RAG generate → %s  query=%r", url, query[:80])
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return {
        "answer": _normalize_answer(data),
        "chunks": _normalize_chunks(data),
    }


async def ingest_file(
    file_bytes: bytes,
    filename: str,
    collection: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Upload a single file to the ingestor-server."""
    url = _url(settings.ingest_url, INGEST_PATH)
    files_payload = {"file": (filename, file_bytes)}
    data_payload: dict[str, Any] = {}
    if collection:
        data_payload["collection"] = collection
    if metadata:
        # Send metadata as JSON string field
        import json
        data_payload["metadata"] = json.dumps(metadata)

    logger.info("Ingest → %s  file=%s  collection=%s", url, filename, collection)
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(url, files=files_payload, data=data_payload)
        resp.raise_for_status()
        return resp.json()


def ingest_file_sync(
    file_bytes: bytes,
    filename: str,
    collection: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synchronous version of ``ingest_file`` (used by CLI scripts)."""
    url = _url(settings.ingest_url, INGEST_PATH)
    files_payload = {"file": (filename, file_bytes)}
    data_payload: dict[str, Any] = {}
    if collection:
        data_payload["collection"] = collection
    if metadata:
        import json
        data_payload["metadata"] = json.dumps(metadata)

    logger.info("Ingest (sync) → %s  file=%s", url, filename)
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(url, files=files_payload, data=data_payload)
        resp.raise_for_status()
        return resp.json()


async def healthcheck_rag() -> dict[str, Any]:
    """Ping the RAG server root or /health."""
    for path in ["/health", "/v1/health", "/"]:
        url = _url(settings.rag_url, path)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(url)
                if resp.status_code < 500:
                    return {"rag_server": "ok", "url": url, "status": resp.status_code}
        except httpx.HTTPError:
            continue
    return {"rag_server": "unreachable", "url": settings.rag_url}


async def healthcheck_ingest() -> dict[str, Any]:
    """Ping the ingestor server root or /health."""
    for path in ["/health", "/v1/health", "/"]:
        url = _url(settings.ingest_url, path)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(url)
                if resp.status_code < 500:
                    return {"ingest_server": "ok", "url": url, "status": resp.status_code}
        except httpx.HTTPError:
            continue
    return {"ingest_server": "unreachable", "url": settings.ingest_url}
