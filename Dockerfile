# ── Stage 1: build & install deps ─────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for building wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
# Create a minimal package so pip can resolve the editable install
COPY app/__init__.py app/__init__.py

RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 2: runtime ─────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="you@example.com"
LABEL description="NVIDIA Admission Agent — Agentic RAG gateway for university admissions"

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Create data directory (mount point)
RUN mkdir -p /app/data/docs

# Non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

# Default env vars (override via docker-compose / .env / -e)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=9000 \
    RAG_URL=http://rag-server:8081 \
    INGEST_URL=http://ingestor-server:8082

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
