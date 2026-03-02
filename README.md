# 🎓 NVIDIA Admission Agent

Agentic RAG chatbot for university admissions **(Fall 2026-2027)** powered by
[NVIDIA RAG Blueprint](https://docs.nvidia.com/ai-enterprise/rag-blueprint/).

The gateway calls **external** NVIDIA Blueprint services (rag-server +
ingestor-server) via HTTP — no blueprint code is embedded.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Client (curl / UI)                                          │
└───────────┬──────────────────────────────────────────────────┘
            │  HTTP :9000
┌───────────▼──────────────────────────────────────────────────┐
│  FastAPI Gateway  (this repo)                                │
│  ├─ /api/v1/search         → RAG search                     │
│  ├─ /api/v1/chat/query     → RAG generate (grounded Q&A)    │
│  ├─ /api/v1/ingest/file    → Upload doc                     │
│  ├─ /api/v1/agent/checklist→ LangGraph checklist flow        │
│  └─ /api/v1/agent/email    → LangGraph email flow            │
└──┬────────────────────┬──────────────────────────────────────┘
   │ HTTP               │ HTTP
┌──▼─────────────┐  ┌───▼──────────────┐
│  rag-server    │  │  ingestor-server  │
│  :8081         │  │  :8082            │
└────────────────┘  └──────────────────┘
```

## Quick Start

### 1. Clone & configure

```bash
git clone <repo-url> && cd nvidia-admission-agent
cp .env.example .env
# Edit .env — set RAG_URL, INGEST_URL, LLM_API_KEY, etc.
```

### 2. Install dependencies (uv recommended)

```bash
uv sync          # or: pip install -e .
```

### 3. Add your documents

Place PDF/Markdown files under `data/docs/` following the convention:

```
data/docs/<university>/<program>/<term>/<file>.pdf
```

Example:

```
data/docs/USC/PhD_CS/Fall_2026/admission.pdf
data/docs/MIT/MS_AI/Fall_2026/requirements.md
```

### 4. Ingest documents

```bash
python scripts/ingest_docs.py
# Preview only:
python scripts/ingest_docs.py --dry-run
```

### 5. Run the server

```bash
uvicorn app.main:app --reload --port 9000
```

### 6. Health check

```bash
# CLI
python scripts/healthcheck.py

# HTTP
curl http://localhost:9000/health
```

---

## API Reference

### `POST /api/v1/search`

Return matching chunks with relevance scores.

```bash
curl -X POST http://localhost:9000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "USC PhD CS admission requirements",
    "top_k": 5
  }'
```

### `POST /api/v1/chat/query`

Grounded Q&A (uses RAG server generate endpoint).

```bash
curl -X POST http://localhost:9000/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What GRE score does MIT require for MS AI?"
  }'
```

### `POST /api/v1/ingest/file`

Upload a single file for ingestion.

```bash
curl -X POST http://localhost:9000/api/v1/ingest/file \
  -F "file=@data/docs/USC/PhD_CS/Fall_2026/admission.pdf" \
  -F "university=USC" \
  -F "program=PhD_CS" \
  -F "term=Fall_2026"
```

### `POST /api/v1/agent/checklist`

Generate a structured admissions checklist (requires `LLM_API_KEY`).

```bash
curl -X POST http://localhost:9000/api/v1/agent/checklist \
  -H "Content-Type: application/json" \
  -d '{
    "university": "USC",
    "program": "PhD_CS",
    "term": "Fall_2026",
    "query": "What documents do I need?"
  }'
```

**Response (JSON):**

```json
{
  "university": "USC",
  "program": "PhD_CS",
  "term": "Fall_2026",
  "required_documents": ["Official transcripts [1]", "Statement of Purpose [1]", "3 Letters of Recommendation [2]"],
  "optional_documents": ["GRE scores [1]"],
  "tests": ["TOEFL/IELTS for international students [2]"],
  "deadlines": [{"description": "Application deadline", "date": "December 15, 2025", "notes": "Per [1]"}],
  "submission_portal": "https://apply.usc.edu",
  "notes": ["Fee waivers available for eligible applicants [2]"],
  "citations": [{"i": 1, "source": "USC/PhD_CS/Fall_2026/admission.pdf", "score": 0.95, "snippet": "..."}]
}
```

### `POST /api/v1/agent/email`

Draft an email to a professor or admissions office (requires `LLM_API_KEY`).

**Professor email:**

```bash
curl -X POST http://localhost:9000/api/v1/agent/email \
  -H "Content-Type: application/json" \
  -d '{
    "recipient_type": "professor",
    "university": "MIT",
    "program": "MS_AI",
    "term": "Fall_2026",
    "professor_name": "Dr. Jane Smith",
    "applicant_name": "Nguyen Van A",
    "applicant_background": "BSc CS from VNU, GPA 3.8, 2 NeurIPS papers",
    "research_interests": "Reinforcement learning, multi-agent systems"
  }'
```

**Admission email:**

```bash
curl -X POST http://localhost:9000/api/v1/agent/email \
  -H "Content-Type: application/json" \
  -d '{
    "recipient_type": "admission",
    "university": "USC",
    "program": "PhD_CS",
    "term": "Fall_2026",
    "applicant_name": "Nguyen Van A",
    "specific_questions": "Is GRE waiver available? What is the transcript evaluation process?"
  }'
```

**Response (JSON):**

```json
{
  "subject": "Prospective PhD Student — Fall 2026 Inquiry",
  "body": "Dear ...",
  "attachments_needed": ["CV", "Unofficial Transcript"],
  "followup_questions": [],
  "citations": [{"i": 1, "source": "...", "score": 0.9, "snippet": "..."}]
}
```

> If required info is missing, `followup_questions` will list what the agent
> needs (e.g. *"What is your full name?"*, *"Which professor would you like to
> contact?"*).

---

## Docker

### Run everything with Docker Compose

```bash
cp .env.example .env   # edit with your keys
docker compose up -d --build
```

This starts **3 services** on a shared `rag-net` network:

| Service | Port | Description |
|---------|------|-------------|
| `admission-agent` | 9000 | FastAPI gateway (this repo) |
| `rag-server` | 8081 | NVIDIA RAG Blueprint — search & generate |
| `ingestor-server` | 8082 | NVIDIA RAG Blueprint — document ingestion |

> **Note:** The `rag-server` and `ingestor-server` images in `docker-compose.yml` are
> **placeholders**. Replace them with your actual NVIDIA RAG Blueprint images, or remove
> them if you run the blueprint separately.

### Run only the gateway (blueprint running externally)

```bash
# Build
docker build -t admission-agent .

# Run
docker run -d --name admission-agent \
  -p 9000:9000 \
  -v ./data/docs:/app/data/docs:ro \
  -e RAG_URL=http://YOUR_RAG_HOST:8081 \
  -e INGEST_URL=http://YOUR_INGEST_HOST:8082 \
  -e LLM_API_KEY=your-key \
  admission-agent
```

### Ingest docs from inside Docker

```bash
docker exec admission-agent python scripts/ingest_docs.py
# or with dry-run:
docker exec admission-agent python scripts/ingest_docs.py --dry-run
```

### Health check

```bash
curl http://localhost:9000/health
```

### Dev mode (hot-reload with mounted source)

```bash
docker run --rm -it \
  -p 9000:9000 \
  -v ./app:/app/app \
  -v ./scripts:/app/scripts \
  -v ./data/docs:/app/data/docs:ro \
  --env-file .env \
  admission-agent \
  uvicorn app.main:app --reload --host 0.0.0.0 --port 9000
```

---

## Configuration

All settings live in `.env` (see `.env.example`):

| Variable               | Default                              | Description                          |
|------------------------|--------------------------------------|--------------------------------------|
| `RAG_URL`              | `http://localhost:8081`              | NVIDIA rag-server base URL           |
| `INGEST_URL`           | `http://localhost:8082`              | NVIDIA ingestor-server base URL      |
| `RAG_SEARCH_PATH`      | `/search`                            | Search endpoint path                 |
| `RAG_GENERATE_PATH`    | `/generate`                          | Generate endpoint path               |
| `INGEST_DOCUMENTS_PATH`| `/documents`                         | Ingest endpoint path                 |
| `RAG_COLLECTION`       | `admissions_fall_2026`               | Default collection name              |
| `LLM_BASE_URL`         | `https://integrate.api.nvidia.com/v1`| OpenAI-compatible LLM endpoint       |
| `LLM_API_KEY`          | *(empty)*                            | API key for agent LLM calls          |
| `LLM_MODEL`            | `meta/llama-3.1-70b-instruct`       | Model name                           |
| `REQUEST_TIMEOUT`      | `60`                                 | HTTP timeout in seconds              |

---

## Repo Structure

```
nvidia-admission-agent/
├─ README.md
├─ .env.example
├─ pyproject.toml
├─ Dockerfile                       ← multi-stage Python 3.12 image
├─ docker-compose.yml               ← gateway + blueprint stubs
├─ .dockerignore
├─ data/
│  └─ docs/                        ← place PDFs here
├─ scripts/
│  ├─ ingest_docs.py               ← batch ingest CLI
│  └─ healthcheck.py               ← ping upstream servers
└─ app/
   ├─ main.py                      ← FastAPI entry point
   ├─ config.py                    ← pydantic-settings
   ├─ schemas.py                   ← all request/response models
   ├─ clients/
   │  └─ nvidia_rag_http.py        ← centralised HTTP client
   ├─ services/
   │  ├─ rag_service.py            ← search & chat orchestration
   │  └─ agent_service.py          ← LangGraph checklist & email flows
   └─ api/
      ├─ __init__.py               ← router aggregation
      ├─ routes_search.py
      ├─ routes_chat.py
      ├─ routes_ingest.py
      └─ routes_agent.py
```

---

## License

MIT
