#!/usr/bin/env python3
"""Quick health-check for upstream NVIDIA RAG + Ingestor servers.

Usage:
    python scripts/healthcheck.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.nvidia_rag_http import healthcheck_ingest, healthcheck_rag  # noqa: E402
from app.config import settings  # noqa: E402


async def main() -> None:
    print("NVIDIA Admission Agent — Health Check")
    print("=" * 50)
    print(f"RAG server URL   : {settings.rag_url}")
    print(f"Ingest server URL: {settings.ingest_url}")
    print(f"LLM configured   : {'Yes' if settings.llm_api_key else 'No'}")
    print()

    rag = await healthcheck_rag()
    ingest = await healthcheck_ingest()

    print(f"RAG server   : {rag}")
    print(f"Ingest server: {ingest}")

    if "unreachable" in str(rag) or "unreachable" in str(ingest):
        print("\n⚠  One or more upstream servers are unreachable.")
        sys.exit(1)
    else:
        print("\n✔  All upstream servers are reachable.")


if __name__ == "__main__":
    asyncio.run(main())
