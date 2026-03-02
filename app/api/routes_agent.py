"""Agent endpoints — checklist & email generation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas import (
    ChecklistRequest,
    ChecklistResponse,
    EmailRequest,
    EmailResponse,
)
from app.services import agent_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent"])


@router.post("/agent/checklist", response_model=ChecklistResponse)
async def agent_checklist(req: ChecklistRequest) -> ChecklistResponse:
    """Return a structured admissions checklist grounded in RAG context."""
    try:
        return await agent_service.run_checklist(req)
    except RuntimeError as exc:
        # LLM not configured
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Checklist agent failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc


@router.post("/agent/email", response_model=EmailResponse)
async def agent_email(req: EmailRequest) -> EmailResponse:
    """Draft an email (professor or admission) grounded in RAG context."""
    try:
        return await agent_service.run_email(req)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Email agent failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
