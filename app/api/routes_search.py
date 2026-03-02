"""POST /api/v1/search — RAG chunk search."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas import SearchRequest, SearchResponse
from app.services import rag_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """Return matching chunks with scores, sources, and citations."""
    try:
        return await rag_service.search(
            query=req.query,
            collection=req.collection,
            top_k=req.top_k,
        )
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=502, detail=f"RAG search error: {exc}") from exc
