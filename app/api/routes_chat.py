"""POST /api/v1/chat/query — grounded RAG Q&A."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas import ChatQueryRequest, ChatQueryResponse
from app.services import rag_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat/query", response_model=ChatQueryResponse)
async def chat_query(req: ChatQueryRequest) -> ChatQueryResponse:
    """Generate a grounded answer using the RAG server."""
    try:
        return await rag_service.chat_query(
            query=req.query,
            collection=req.collection,
            top_k=req.top_k,
        )
    except Exception as exc:
        logger.exception("Chat query failed")
        raise HTTPException(status_code=502, detail=f"RAG generate error: {exc}") from exc
