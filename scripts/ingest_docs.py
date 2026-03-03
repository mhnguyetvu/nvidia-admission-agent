#!/usr/bin/env python3
"""Batch-ingest all PDFs and Markdown files under ``data/docs/`` into local ChromaDB.

Directory convention
--------------------
    data/docs/<university>/<program>/<term>/<file>.pdf

Metadata extracted automatically:
    university, program, term, path, filename, ext, ingested_at

Usage
-----
    python scripts/ingest_docs.py              # ingest everything
    python scripts/ingest_docs.py --dry-run    # preview what would be uploaded
    python scripts/ingest_docs.py --dir data/docs/SNU   # only SNU
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.local_chroma import ingest_file_sync
from app.config import settings  # noqa: E402

SUPPORTED_EXTENSIONS = {".pdf", ".md"}


def _extract_metadata(filepath: Path, docs_root: Path) -> dict[str, str]:
    """Derive metadata from the file's relative path.

    Expected layout: ``<university>/<program>/<term>/<file>.ext``
    """
    rel = filepath.relative_to(docs_root)
    parts = rel.parts  # e.g. ("SNU", "MS_ML", "Fall_2026", "admission.pdf")

    meta: dict[str, str] = {
        "path": str(rel),
        "filename": filepath.name,
        "ext": filepath.suffix.lstrip("."),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    if len(parts) >= 1:
        meta["university"] = parts[0]
    if len(parts) >= 2:
        meta["program"] = parts[1]
    if len(parts) >= 3:
        meta["term"] = parts[2]

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into local ChromaDB")
    parser.add_argument(
        "--dir",
        type=str,
        default="data/docs",
        help="Root directory to scan (default: data/docs)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="",
        help=f"Target collection (default from .env: {settings.rag_collection})",
    )
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    args = parser.parse_args()

    docs_root = Path(args.dir).resolve()
    if not docs_root.is_dir():
        print(f"ERROR: Directory not found: {docs_root}")
        sys.exit(1)

    collection = args.collection or settings.rag_collection
    files = sorted(
        f
        for f in docs_root.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        print(f"No .pdf / .md files found under {docs_root}")
        sys.exit(0)

    print(f"Backend: Local ChromaDB")
    print(f"Found {len(files)} file(s) under {docs_root}")
    print(f"Target collection: {collection}")
    print(f"ChromaDB dir: {Path(settings.chroma_dir).resolve()}")
    print()

    success = 0
    failed = 0

    for f in files:
        meta = _extract_metadata(f, docs_root)
        label = meta.get("path", f.name)

        if args.dry_run:
            print(f"  [DRY-RUN] {label}  metadata={meta}")
            continue

        try:
            resp = ingest_file_sync(
                file_bytes=f.read_bytes(),
                filename=f.name,
                collection=collection,
                metadata=meta,
            )
            chunks_info = resp.get("chunks_created", "?")
            print(f"  OK  {label}  ({chunks_info} chunks)")
            success += 1
        except Exception as exc:
            print(f"  FAIL  {label}  ERROR: {exc}")
            failed += 1

    if not args.dry_run:
        print(f"\nDone. {success} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
