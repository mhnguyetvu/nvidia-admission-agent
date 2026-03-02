"""FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --port 9000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.clients import nvidia_rag_http as rag_client
from app.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

app = FastAPI(
    title="NVIDIA Admission Agent",
    description=(
        "Agentic RAG gateway for university admissions (Fall 2026-2027). "
        "Powered by NVIDIA RAG Blueprint."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
async def health() -> dict:
    """Quick health check — also pings upstream RAG / ingest servers."""
    rag = await rag_client.healthcheck_rag()
    ingest = await rag_client.healthcheck_ingest()
    llm_configured = bool(settings.llm_api_key)
    return {
        "status": "ok",
        "rag_server": rag,
        "ingest_server": ingest,
        "llm_configured": llm_configured,
        "collection": settings.rag_collection,
    }
