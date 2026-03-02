"""Pydantic v2 schemas for API request / response models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Shared ─────────────────────────────────────────────────────────


class Citation(BaseModel):
    """A single citation referencing a retrieved chunk."""

    i: int = Field(..., description="1-based citation index")
    source: str = Field("", description="Source file path or URL")
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = Field(0.0, description="Retrieval similarity score")
    snippet: str = Field("", description="Excerpt from the chunk")


# ── Search ─────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str
    collection: str = ""
    top_k: int = Field(default=5, ge=1, le=50)


class Chunk(BaseModel):
    text: str
    score: float = 0.0
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    chunks: list[Chunk]
    citations: list[Citation]


# ── Chat ───────────────────────────────────────────────────────────


class ChatQueryRequest(BaseModel):
    query: str
    collection: str = ""
    top_k: int = Field(default=5, ge=1, le=50)


class ChatQueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]


# ── Ingest ─────────────────────────────────────────────────────────


class IngestResponse(BaseModel):
    status: str
    filename: str
    collection: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


# ── Agent: Checklist ───────────────────────────────────────────────


class DeadlineItem(BaseModel):
    description: str
    date: str = ""
    notes: str = ""


class ChecklistRequest(BaseModel):
    university: str = ""
    program: str = ""
    term: str = ""
    query: str = ""
    collection: str = ""


class ChecklistResponse(BaseModel):
    university: str
    program: str
    term: str
    required_documents: list[str]
    optional_documents: list[str]
    tests: list[str]
    deadlines: list[DeadlineItem]
    submission_portal: str = ""
    notes: list[str]
    citations: list[Citation]


# ── Agent: Email ───────────────────────────────────────────────────


class RecipientType(str, Enum):
    professor = "professor"
    admission = "admission"


class EmailRequest(BaseModel):
    recipient_type: RecipientType
    university: str = ""
    program: str = ""
    term: str = ""
    professor_name: str = ""
    applicant_name: str = ""
    applicant_background: str = ""
    research_interests: str = ""
    specific_questions: str = ""
    collection: str = ""
    query: str = ""


class EmailResponse(BaseModel):
    subject: str
    body: str
    attachments_needed: list[str]
    followup_questions: list[str]
    citations: list[Citation]
