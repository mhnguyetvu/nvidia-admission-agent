"""API routes package — register all sub-routers here."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.routes_agent import router as agent_router
from app.api.routes_chat import router as chat_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_search import router as search_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(search_router)
api_router.include_router(chat_router)
api_router.include_router(ingest_router)
api_router.include_router(agent_router)
