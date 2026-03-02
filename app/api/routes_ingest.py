"""POST /api/v1/ingest/file — upload a single file for ingestion."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.clients import nvidia_rag_http as rag_client
from app.config import settings
from app.schemas import IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    collection: str = Form(default=""),
    university: str = Form(default=""),
    program: str = Form(default=""),
    term: str = Form(default=""),
) -> IngestResponse:
    """Upload a single file (PDF/MD) to the NVIDIA ingestor-server."""
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

    try:
        resp = await rag_client.ingest_file(
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
