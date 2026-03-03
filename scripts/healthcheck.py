#!/usr/bin/env python3
"""Quick health-check for local ChromaDB backend.

Usage:
    python scripts/healthcheck.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.local_chroma import healthcheck_ingest, healthcheck_rag  # noqa: E402
from app.config import settings  # noqa: E402


async def main() -> None:
    print("Admission Agent — Health Check")
    print("=" * 50)
    print(f"ChromaDB dir: {settings.chroma_dir}")
    print(f"LLM configured   : {'Yes' if settings.llm_api_key else 'No'}")
    print()

    rag = await healthcheck_rag()
    ingest = await healthcheck_ingest()

    print(f"ChromaDB RAG   : {rag}")
    print(f"ChromaDB Ingest: {ingest}")

    if "unreachable" in str(rag) or "unreachable" in str(ingest):
        print("\n⚠  ChromaDB backend is not accessible.")
        sys.exit(1)
    else:
        print("\n✔  ChromaDB backend is healthy.")


if __name__ == "__main__":
    asyncio.run(main())
