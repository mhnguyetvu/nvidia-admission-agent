"""FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload --port 9000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_router
from app.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NVIDIA Admission Agent",
    description=(
        "Agentic RAG gateway for university admissions (Fall 2026-2027). "
        "Powered by NVIDIA RAG Blueprint or local ChromaDB."
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


@app.on_event("startup")
async def startup_log() -> None:
    logger.info("Backend mode: %s", settings.backend)
    if settings.backend == "local":
        logger.info("ChromaDB dir: %s", settings.chroma_dir)
        logger.info("Embedding model: %s", settings.embedding_model)
    else:
        logger.info("RAG URL: %s", settings.rag_url)
        logger.info("Ingest URL: %s", settings.ingest_url)
    logger.info("LLM configured: %s", bool(settings.llm_api_key))


@app.get("/health")
async def health() -> dict:
    """Quick health check — pings the active backend."""
    if settings.backend == "local":
        from app.clients import local_chroma as client
    else:
        from app.clients import nvidia_rag_http as client

    rag = await client.healthcheck_rag()
    ingest = await client.healthcheck_ingest()
    llm_configured = bool(settings.llm_api_key)

    return {
        "status": "ok",
        "backend": settings.backend,
        "rag": rag,
        "ingest": ingest,
        "llm_configured": llm_configured,
        "collection": settings.rag_collection,
    }
