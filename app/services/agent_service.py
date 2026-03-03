"""Agentic service — LangGraph-powered flows for checklist & email generation.

Architecture:
  1.  Always RAG-search first (tool-first).
  2.  Router classifies intent → checklist flow | email flow.
  3.  LLM synthesises grounded answer from retrieved context.
  4.  Email flow detects missing info and returns followup_questions[].
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from app.config import settings
from app.schemas import (
    ChecklistRequest,
    ChecklistResponse,
    Citation,
    DeadlineItem,
    EmailRequest,
    EmailResponse,
    RecipientType,
)

logger = logging.getLogger(__name__)

# ── LLM helper ────────────────────────────────────────────────────


def _get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance configured from .env."""
    if not settings.llm_api_key:
        raise RuntimeError(
            "LLM_API_KEY is not set. Please configure it in your .env file to use agent endpoints."
        )
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0.1,
        max_tokens=4096,
    )


def _build_citations(chunks: list[dict[str, Any]]) -> list[Citation]:
    return [
        Citation(
            i=idx + 1,
            source=c.get("source", ""),
            metadata=c.get("metadata", {}),
            score=c.get("score", 0.0),
            snippet=c.get("text", "")[:300],
        )
        for idx, c in enumerate(chunks)
    ]


def _context_block(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    parts: list[str] = []
    for idx, c in enumerate(chunks, 1):
        source = c.get("source", "unknown")
        meta = c.get("metadata", {})
        uni = meta.get("university", "")
        prog = meta.get("program", "")
        text = c.get("text", "")
        header = f"[{idx}] Source: {source}"
        if uni:
            header += f" | University: {uni}"
        if prog:
            header += f" | Program: {prog}"
        parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(parts)


# ── Prompts ────────────────────────────────────────────────────────

CHECKLIST_SYSTEM_PROMPT = """\
You are an expert university admissions advisor. Your ONLY source of truth is
the CONTEXT below, which comes from official PDF documents.

RULES:
- Extract information ONLY from the provided context.
- If a piece of information is not found, write "Not found in provided documents — please check [source name]" in the relevant field.
- Reference citations using [1], [2], etc. corresponding to the context chunk numbers.
- Output MUST be valid JSON matching the schema exactly (no markdown fences).

OUTPUT JSON SCHEMA:
{
  "university": "<string>",
  "program": "<string>",
  "term": "<string>",
  "required_documents": ["<string with [citation]>", ...],
  "optional_documents": ["<string with [citation]>", ...],
  "tests": ["<string with [citation]>", ...],
  "deadlines": [{"description": "<string>", "date": "<string>", "notes": "<string>"}],
  "submission_portal": "<string or empty>",
  "notes": ["<string with [citation]>", ...]
}
"""

PROFESSOR_EMAIL_SYSTEM_PROMPT = """\
You are a professional academic email writer. Draft a concise email from a
prospective graduate student to a professor.

RULES:
- Use ONLY the context provided to reference the program and professor's work.
- The email must: express specific research fit, ask about availability for
  the indicated term, mention attaching a CV.
- Keep the email SHORT (under 200 words body).
- Reference citations as [1], [2] where facts come from context.
- If critical information is missing (applicant name, professor name, background,
  research interests), list them in "followup_questions".

OUTPUT JSON SCHEMA:
{
  "subject": "<string>",
  "body": "<string>",
  "attachments_needed": ["CV", ...],
  "followup_questions": ["<string>", ...],
}
"""

ADMISSION_EMAIL_SYSTEM_PROMPT = """\
You are a professional academic email writer. Draft a polite email from a
prospective student to a university admissions office.

RULES:
- Use ONLY the context to cite specific requirements, deadlines, and document
  checklists from the PDFs.
- Ask clearly about transcript requirements, score waivers, deadlines, etc.
- Reference citations as [1], [2] for each claim.
- If critical information is missing, list them in "followup_questions".

OUTPUT JSON SCHEMA:
{
  "subject": "<string>",
  "body": "<string>",
  "attachments_needed": ["<string>", ...],
  "followup_questions": ["<string>", ...],
}
"""


# ── LangGraph state ───────────────────────────────────────────────


class AgentState(TypedDict, total=False):
    intent: str
    query: str
    collection: str
    top_k: int
    chunks: list[dict[str, Any]]
    citations: list[Citation]
    request_data: dict[str, Any]
    llm_output: str
    result: dict[str, Any]
    error: str


# ── Graph nodes ────────────────────────────────────────────────────


async def rag_search_node(state: AgentState) -> AgentState:
    """Always search RAG first."""
    from app.clients import local_chroma as rag_client

    query = state.get("query", "")
    collection = state.get("collection", "") or settings.rag_collection
    top_k = state.get("top_k", 8)
    try:
        chunks = await rag_client.search(query=query, collection=collection, top_k=top_k)
    except Exception as exc:
        logger.error("RAG search failed: %s", exc)
        chunks = []
    citations = _build_citations(chunks)
    return {**state, "chunks": chunks, "citations": citations}


async def checklist_synth_node(state: AgentState) -> AgentState:
    """Synthesise a checklist from retrieved chunks."""
    llm = _get_llm()
    ctx = _context_block(state.get("chunks", []))
    req = state.get("request_data", {})
    user_msg = (
        f"University: {req.get('university', 'any')}\n"
        f"Program: {req.get('program', 'any')}\n"
        f"Term: {req.get('term', 'any')}\n"
        f"Additional query: {req.get('query', '')}\n\n"
        f"CONTEXT:\n{ctx}"
    )
    resp = await llm.ainvoke([
        SystemMessage(content=CHECKLIST_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])
    return {**state, "llm_output": resp.content}


async def email_synth_node(state: AgentState) -> AgentState:
    """Synthesise an email draft from retrieved chunks."""
    llm = _get_llm()
    ctx = _context_block(state.get("chunks", []))
    req = state.get("request_data", {})
    recipient_type = req.get("recipient_type", "admission")
    system_prompt = (
        PROFESSOR_EMAIL_SYSTEM_PROMPT
        if recipient_type == "professor"
        else ADMISSION_EMAIL_SYSTEM_PROMPT
    )
    user_msg = (
        f"Recipient type: {recipient_type}\n"
        f"University: {req.get('university', '')}\n"
        f"Program: {req.get('program', '')}\n"
        f"Term: {req.get('term', '')}\n"
        f"Professor: {req.get('professor_name', '')}\n"
        f"Applicant name: {req.get('applicant_name', '')}\n"
        f"Background: {req.get('applicant_background', '')}\n"
        f"Research interests: {req.get('research_interests', '')}\n"
        f"Specific questions: {req.get('specific_questions', '')}\n\n"
        f"CONTEXT:\n{ctx}"
    )
    resp = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ])
    return {**state, "llm_output": resp.content}


def _safe_parse_json(raw: str) -> dict[str, Any]:
    """Try to extract a JSON object from LLM output (may contain markdown fences)."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find first { ... } block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def build_checklist_result(state: AgentState) -> AgentState:
    """Parse LLM checklist JSON and build result."""
    parsed = _safe_parse_json(state.get("llm_output", "{}"))
    req = state.get("request_data", {})
    citations = state.get("citations", [])
    result = {
        "university": parsed.get("university", req.get("university", "")),
        "program": parsed.get("program", req.get("program", "")),
        "term": parsed.get("term", req.get("term", "")),
        "required_documents": parsed.get("required_documents", []),
        "optional_documents": parsed.get("optional_documents", []),
        "tests": parsed.get("tests", []),
        "deadlines": parsed.get("deadlines", []),
        "submission_portal": parsed.get("submission_portal", ""),
        "notes": parsed.get("notes", []),
        "citations": [c.model_dump() for c in citations],
    }
    return {**state, "result": result}


def build_email_result(state: AgentState) -> AgentState:
    """Parse LLM email JSON and build result."""
    parsed = _safe_parse_json(state.get("llm_output", "{}"))
    citations = state.get("citations", [])

    # Detect missing info → followup_questions
    req = state.get("request_data", {})
    followups: list[str] = list(parsed.get("followup_questions", []))
    recipient_type = req.get("recipient_type", "admission")
    if recipient_type == "professor":
        if not req.get("applicant_name"):
            followups.append("What is your full name?")
        if not req.get("professor_name"):
            followups.append("Which professor would you like to contact?")
        if not req.get("applicant_background"):
            followups.append("Please describe your academic background briefly.")
        if not req.get("research_interests"):
            followups.append("What are your research interests?")
    else:
        if not req.get("applicant_name"):
            followups.append("What is your full name?")

    # Deduplicate
    seen: set[str] = set()
    unique_followups: list[str] = []
    for q in followups:
        if q not in seen:
            unique_followups.append(q)
            seen.add(q)

    result = {
        "subject": parsed.get("subject", ""),
        "body": parsed.get("body", ""),
        "attachments_needed": parsed.get("attachments_needed", []),
        "followup_questions": unique_followups,
        "citations": [c.model_dump() for c in citations],
    }
    return {**state, "result": result}


# ── Graph builders ─────────────────────────────────────────────────


def _build_checklist_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("rag_search", rag_search_node)
    g.add_node("checklist_synth", checklist_synth_node)
    g.add_node("build_result", build_checklist_result)
    g.set_entry_point("rag_search")
    g.add_edge("rag_search", "checklist_synth")
    g.add_edge("checklist_synth", "build_result")
    g.add_edge("build_result", END)
    return g


def _build_email_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("rag_search", rag_search_node)
    g.add_node("email_synth", email_synth_node)
    g.add_node("build_result", build_email_result)
    g.set_entry_point("rag_search")
    g.add_edge("rag_search", "email_synth")
    g.add_edge("email_synth", "build_result")
    g.add_edge("build_result", END)
    return g


# Compile once
_checklist_app = _build_checklist_graph().compile()
_email_app = _build_email_graph().compile()


# ── Public API ─────────────────────────────────────────────────────


async def run_checklist(req: ChecklistRequest) -> ChecklistResponse:
    """Execute the checklist agentic flow."""
    search_query = " ".join(
        filter(None, [req.university, req.program, req.term, req.query, "admission requirements checklist deadlines"])
    )
    initial_state: AgentState = {
        "intent": "checklist",
        "query": search_query,
        "collection": req.collection or settings.rag_collection,
        "top_k": 8,
        "request_data": req.model_dump(),
    }
    final = await _checklist_app.ainvoke(initial_state)
    result = final.get("result", {})

    # Convert deadline dicts → DeadlineItem
    deadlines = []
    for d in result.get("deadlines", []):
        if isinstance(d, dict):
            deadlines.append(DeadlineItem(**d))
        elif isinstance(d, DeadlineItem):
            deadlines.append(d)

    # Convert citation dicts → Citation
    citations = []
    for c in result.get("citations", []):
        if isinstance(c, dict):
            citations.append(Citation(**c))
        elif isinstance(c, Citation):
            citations.append(c)

    return ChecklistResponse(
        university=result.get("university", req.university),
        program=result.get("program", req.program),
        term=result.get("term", req.term),
        required_documents=result.get("required_documents", []),
        optional_documents=result.get("optional_documents", []),
        tests=result.get("tests", []),
        deadlines=deadlines,
        submission_portal=result.get("submission_portal", ""),
        notes=result.get("notes", []),
        citations=citations,
    )


async def run_email(req: EmailRequest) -> EmailResponse:
    """Execute the email agentic flow."""
    search_query = " ".join(
        filter(
            None,
            [
                req.university,
                req.program,
                req.term,
                req.query,
                "admission contact",
                req.professor_name,
                req.specific_questions,
            ],
        )
    )
    initial_state: AgentState = {
        "intent": "email",
        "query": search_query,
        "collection": req.collection or settings.rag_collection,
        "top_k": 8,
        "request_data": req.model_dump(mode="json"),
    }
    final = await _email_app.ainvoke(initial_state)
    result = final.get("result", {})

    citations = []
    for c in result.get("citations", []):
        if isinstance(c, dict):
            citations.append(Citation(**c))
        elif isinstance(c, Citation):
            citations.append(c)

    return EmailResponse(
        subject=result.get("subject", ""),
        body=result.get("body", ""),
        attachments_needed=result.get("attachments_needed", []),
        followup_questions=result.get("followup_questions", []),
        citations=citations,
    )
