"""Centralized configuration via pydantic-settings + .env file."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # ── Backend mode ───────────────────────────────────────────
    backend: Literal["local", "nvidia"] = "local"

    # ── NVIDIA RAG Blueprint (only used when backend=nvidia) ───
    rag_url: str = "http://localhost:8081"
    ingest_url: str = "http://localhost:8082"

    rag_search_path: str = "/search"
    rag_generate_path: str = "/generate"
    ingest_documents_path: str = "/documents"

    # ── Shared RAG settings ────────────────────────────────────
    rag_collection: str = "admissions_fall_2026"

    # ── Local mode settings ────────────────────────────────────
    chroma_dir: str = "./chroma_data"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── LLM (OpenAI-compatible) ────────────────────────────────
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_api_key: str = ""
    llm_model: str = "meta/llama-3.1-70b-instruct"

    # ── General ────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 9000
    request_timeout: int = 60
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
