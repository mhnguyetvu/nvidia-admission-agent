"""Local RAG backend using ChromaDB + VNPay BGE m3 Embedding + PyMuPDF.

Drop-in replacement for nvidia_rag_http.py — same function signatures.
Uses VNPay embedding API (BGE m3) for vector generation.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
import fitz  # PyMuPDF
import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# ── Lazy singletons ───────────────────────────────────────────────

_chroma_client: chromadb.ClientAPI | None = None


def _get_chroma() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        chroma_path = Path(settings.chroma_dir).resolve()
        chroma_path.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        logger.info("ChromaDB initialised at %s", chroma_path)
    return _chroma_client


async def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Call VNPay BGE m3 embedding API."""
    url = settings.embedding_base_url
    headers = {
        "Authorization": f"Bearer {settings.embedding_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.embedding_model,
        "input": texts,
    }
    
    logger.debug("Calling embedding API for %d texts", len(texts))
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # Extract embeddings from response
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings
    except Exception as exc:
        logger.error("Embedding API error: %s", exc)
        raise


def _get_collection(name: str = "") -> chromadb.Collection:
    col_name = name or settings.rag_collection

    class _EmbFn(chromadb.EmbeddingFunction):
        """Sync wrapper for async VNPay embedding API."""
        def __call__(self, input: list[str]) -> list[list[float]]:
            import asyncio
            import httpx
            
            # Sync version of embedding call
            url = settings.embedding_base_url
            headers = {
                "Authorization": f"Bearer {settings.embedding_api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": settings.embedding_model,
                "input": input,
            }
            
            try:
                with httpx.Client(timeout=60) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    embeddings = [item["embedding"] for item in data["data"]]
                    return embeddings
            except Exception as exc:
                logger.error("Embedding API error: %s", exc)
                raise

    return _get_chroma().get_or_create_collection(
        name=col_name,
        embedding_function=_EmbFn(),
        metadata={"hnsw:space": "cosine"},
    )


# ── PDF / Markdown parsing ────────────────────────────────────────


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def _extract_text_from_md(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def _extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext == "pdf":
        return _extract_text_from_pdf(file_bytes)
    elif ext in ("md", "markdown", "txt"):
        return _extract_text_from_md(file_bytes)
    else:
        # Try as text
        return file_bytes.decode("utf-8", errors="replace")


def _chunk_text(text: str, chunk_size: int = 0, overlap: int = 0) -> list[str]:
    """Split text into overlapping chunks by character count."""
    cs = chunk_size or settings.chunk_size
    ov = overlap or settings.chunk_overlap

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + cs
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - ov
        if start >= len(text):
            break
    return chunks


# ── Public API (mirrors nvidia_rag_http signatures) ───────────────


async def search(
    query: str,
    collection: str = "",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Query ChromaDB and return normalised chunk dicts."""
    col = _get_collection(collection)

    if col.count() == 0:
        logger.warning("Collection '%s' is empty — no results.", col.name)
        return []

    results = col.query(query_texts=[query], n_results=min(top_k, col.count()))

    chunks: list[dict[str, Any]] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        # ChromaDB cosine distance: 0 = identical, 2 = opposite
        # Convert to similarity score: 1 - (dist/2)
        score = round(1.0 - (dist / 2.0), 4)
        chunks.append({
            "text": doc,
            "score": score,
            "source": meta.get("source", ""),
            "metadata": meta,
        })

    logger.info("Local search: query=%r → %d chunks", query[:60], len(chunks))
    return chunks


async def generate(
    query: str,
    collection: str = "",
    top_k: int = 5,
) -> dict[str, Any]:
    """Search + call LLM to generate Vietnamese answer."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    
    chunks = await search(query=query, collection=collection, top_k=top_k)
    
    if not chunks:
        return {
            "answer": "Xin lỗi, tôi không tìm thấy thông tin liên quan trong tài liệu.",
            "chunks": [],
        }
    
    # Build context from top chunks
    context = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks[:3]))
    
    # Vietnamese system prompt
    system_prompt = """Bạn là một trợ lý tư vấn tuyển sinh thạc sĩ chuyên nghiệp. 
Nhiệm vụ của bạn là trả lời câu hỏi bằng tiếng Việt dựa trên tài liệu được cung cấp.

QUAN TRỌNG:
- Trả lời HOÀN TOÀN bằng tiếng Việt
- Chỉ sử dụng thông tin từ tài liệu được cung cấp
- Nếu không có thông tin, hãy nói "Tôi không tìm thấy thông tin này trong tài liệu"
- Trả lời ngắn gọn, rõ ràng và chuyên nghiệp
- Trích dẫn nguồn bằng số [1], [2], [3] khi cần thiết"""

    user_prompt = f"""Dựa trên các tài liệu sau:

{context}

Câu hỏi: {query}

Hãy trả lời bằng tiếng Việt một cách chính xác và đầy đủ."""

    # Call LLM
    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0,
        max_tokens=2048,
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    
    response = await llm.ainvoke(messages)
    answer = response.content.strip()
    
    logger.info("LLM generated Vietnamese answer for query: %r", query[:60])
    
    return {
        "answer": answer,
        "chunks": chunks,
    }


async def ingest_file(
    file_bytes: bytes,
    filename: str,
    collection: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Parse file, chunk, embed, store in ChromaDB."""
    text = _extract_text(file_bytes, filename)
    if not text.strip():
        return {"status": "warning", "message": f"No text extracted from {filename}"}

    chunks = _chunk_text(text)
    if not chunks:
        return {"status": "warning", "message": f"No chunks created from {filename}"}

    col = _get_collection(collection)
    base_meta = metadata or {}
    base_meta.setdefault("source", filename)
    base_meta.setdefault("ingested_at", datetime.now(timezone.utc).isoformat())

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        chunk_id = f"{filename}::chunk_{i}"
        chunk_meta = {**base_meta, "chunk_index": i, "total_chunks": len(chunks)}
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append(chunk_meta)

    col.upsert(ids=ids, documents=documents, metadatas=metadatas)

    logger.info(
        "Ingested %s → %d chunks into collection '%s'",
        filename, len(chunks), col.name,
    )
    return {
        "status": "ok",
        "filename": filename,
        "chunks_created": len(chunks),
        "collection": col.name,
        "text_length": len(text),
    }


def ingest_file_sync(
    file_bytes: bytes,
    filename: str,
    collection: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Synchronous version for CLI scripts."""
    import asyncio
    return asyncio.run(ingest_file(file_bytes, filename, collection, metadata))


async def healthcheck_rag() -> dict[str, Any]:
    """Local mode is always 'ok'."""
    col = _get_collection()
    return {
        "backend": "local (ChromaDB)",
        "status": "ok",
        "collection": col.name,
        "document_count": col.count(),
        "chroma_dir": settings.chroma_dir,
    }


async def healthcheck_ingest() -> dict[str, Any]:
    """Local mode — ingest is built-in."""
    return {
        "backend": "local (PyMuPDF + ChromaDB)",
        "status": "ok",
    }
