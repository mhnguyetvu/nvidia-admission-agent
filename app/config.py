"""Centralized configuration via pydantic-settings + .env file."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # ── Local ChromaDB settings ────────────────────────────────
    rag_collection: str = "admissions_fall_2026"
    chroma_dir: str = "./chroma_data"
    embedding_model: str = "v_search"  # BGE m3 from VNPay
    embedding_base_url: str = "https://genai.vnpay.vn/aigateway/embed/v1/embeddings"
    embedding_api_key: str = ""
    chunk_size: int = 500
    chunk_overlap: int = 50

    # ── LLM (OpenAI-compatible) ────────────────────────────────
    llm_base_url: str = "https://genai.vnpay.vn/aigateway/llm_glm_air/v1"
    llm_api_key: str = ""
    llm_model: str = "v_air45"  # GLM 4.5 Air 110B

    # ── General ────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 9000
    request_timeout: int = 60
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
