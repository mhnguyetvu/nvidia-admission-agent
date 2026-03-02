"""POST /api/v1/ingest/file — upload a single file for ingestion."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.config import settings
from app.schemas import IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])


def _get_ingest_client():
    """Return the correct client module for ingestion."""
    if settings.backend == "local":
        from app.clients import local_chroma as client
    else:
        from app.clients import nvidia_rag_http as client
    return client


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    collection: str = Form(default=""),
    university: str = Form(default=""),
    program: str = Form(default=""),
    term: str = Form(default=""),
) -> IngestResponse:
    """Upload a single file (PDF/MD) for ingestion."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    col = collection or settings.rag_collection
    content = await file.read()
    metadata = {
        "university": university,
        "program": program,
        "term": term,
        "filename": file.filename,
        "ext": file.filename.rsplit(".", 1)[-1] if "." in file.filename else "",
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata = {k: v for k, v in metadata.items() if v}

    client = _get_ingest_client()
    try:
        resp = await client.ingest_file(
            file_bytes=content,
            filename=file.filename,
            collection=col,
            metadata=metadata,
        )
    except Exception as exc:
        logger.exception("Ingest failed for %s", file.filename)
        raise HTTPException(status_code=502, detail=f"Ingest error: {exc}") from exc

    return IngestResponse(
        status="ok",
        filename=file.filename,
        collection=col,
        metadata=metadata,
        message=str(resp),
    )
