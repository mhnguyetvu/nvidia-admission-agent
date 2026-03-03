# 🎓 Admission Agent

Agentic RAG chatbot for university admissions **(Fall 2026-2027)** powered by **VNPay GenAI** and local ChromaDB.

The application uses:
- **VNPay GLM 4.5 Air 110B** for LLM generation (Vietnamese + English support)
- **VNPay BGE m3** for semantic embeddings
- **Local ChromaDB** for vector storage and retrieval
- **FastAPI** for REST API
- **LangGraph** for agent workflows
- **Simple HTML chatbot UI** for interactive Q&A

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Client (curl / UI)                                          │
└───────────┬──────────────────────────────────────────────────┘
            │  HTTP :9000
┌───────────▼──────────────────────────────────────────────────┐
│  FastAPI Gateway  (this repo)                                │
│  ├─ /api/v1/search         → ChromaDB search                │
│  ├─ /api/v1/chat/query     → ChromaDB RAG + LLM             │
│  ├─ /api/v1/ingest/file    → Upload doc to ChromaDB         │
│  ├─ /api/v1/agent/checklist→ LangGraph checklist flow        │
│  └─ /api/v1/agent/email    → LangGraph email flow            │
└──┬────────────────────────────────────────────────────────────┘
   │
┌──▼──────────────────┐
│  Local ChromaDB     │
│  (./chroma_data)    │
└─────────────────────┘
```

## Quick Start

### 1. Clone & configure

```bash
git clone <repo-url> && cd admission-agent
cp .env.example .env
# Edit .env — set LLM_API_KEY for agent features (optional)
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
python -m uvicorn app.main:app --port 7777 --host 0.0.0.0
```

### 6. Open the chatbot UI (optional)

Serve the chatbot interface:

```bash
python -m http.server 8080
```

Then open in your browser: **http://localhost:8080/chatbot.html**

The chatbot will:
- ✅ Respond in **Vietnamese** by default
- 🔍 Search your ingested PDFs with semantic embeddings
- 💬 Generate contextual answers using VNPay GLM 4.5 Air
- 📚 Show source citations with relevance scores

### 7. Health check

```bash
# CLI
python scripts/healthcheck.py

# HTTP
curl http://localhost:7777/health
```

---

## API Reference

### `POST /api/v1/search`

Return matching chunks with relevance scores from ChromaDB.

```bash
curl -X POST http://localhost:7777/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "USC PhD CS admission requirements",
    "top_k": 5
  }'
```

### `POST /api/v1/chat/query`

Grounded Q&A using ChromaDB retrieval + VNPay GLM 4.5 Air LLM generation.

**Responds in Vietnamese by default** based on ingested documents.

```bash
curl -X POST http://localhost:7777/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Yêu cầu tuyển sinh là gì?"
  }'

# Or in English:
curl -X POST http://localhost:7777/api/v1/chat/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the admission requirements?"
  }'
```

### `POST /api/v1/ingest/file`

Upload a single file for ingestion into ChromaDB.

```bash
curl -X POST http://localhost:7777/api/v1/ingest/file \
  -F "file=@data/docs/SNU/MS_ML/Fall_2026/admission.pdf" \
  -F "university=SNU" \
  -F "program=MS_ML" \
  -F "term=Fall_2026"
```

### `POST /api/v1/agent/checklist`

Generate a structured admissions checklist (requires `LLM_API_KEY`).

```bash
curl -X POST http://localhost:7777/api/v1/agent/checklist \
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
curl -X POST http://localhost:7777/api/v1/agent/email \
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
curl -X POST http://localhost:7777/api/v1/agent/email \
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

### Run with Docker Compose

```bash
cp .env.example .env   # edit with your API key (optional for agent features)
docker compose up -d --build
```

This starts the **admission-agent** service on a local network with mounted ChromaDB storage:

| Service | Port | Description |
|---------|------|-------------|
| `admission-agent` | 9000 | FastAPI gateway with local ChromaDB |

### Run only the gateway (manual ChromaDB)

```bash
# Build
docker build -t admission-agent .

# Run
docker run -d --name admission-agent \
  -p 9000:9000 \
  -v ./data/docs:/app/data/docs:ro \
  -v ./chroma_data:/app/chroma_data \
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
  -v ./chroma_data:/app/chroma_data \
  --env-file .env \
  admission-agent \
  uvicorn app.main:app --reload --host 0.0.0.0 --port 9000
```

---

## Configuration

All settings live in `.env` (see `.env.example`):

| Variable               | Default                                          | Description                          |
|------------------------|--------------------------------------------------|--------------------------------------|
| `RAG_COLLECTION`       | `admissions_fall_2026`                           | ChromaDB collection name             |
| `CHROMA_DIR`           | `./chroma_data`                                  | Local ChromaDB storage directory     |
| `EMBEDDING_MODEL`      | `v_search`                                       | VNPay BGE m3 embedding model         |
| `EMBEDDING_BASE_URL`   | `https://genai.vnpay.vn/aigateway/embed/v1/embeddings` | VNPay embedding API endpoint  |
| `EMBEDDING_API_KEY`    | *(required)*                                     | VNPay JWT bearer token               |
| `CHUNK_SIZE`           | `500`                                            | Text chunk size in characters        |
| `CHUNK_OVERLAP`        | `50`                                             | Character overlap between chunks     |
| `LLM_BASE_URL`         | `https://genai.vnpay.vn/aigateway/llm_glm_air/v1`| VNPay GLM 4.5 Air endpoint          |
| `LLM_API_KEY`          | *(required)*                                     | VNPay JWT bearer token               |
| `LLM_MODEL`            | `v_air45`                                        | GLM 4.5 Air 110B model               |
| `REQUEST_TIMEOUT`      | `60`                                             | HTTP timeout in seconds              |

---

## Repo Structure

```
admission-agent/
├─ README.md
├─ .env.example
├─ pyproject.toml
├─ Dockerfile                       ← multi-stage Python 3.12 image
├─ docker-compose.yml               ← local-only gateway setup
├─ .dockerignore
├─ data/
│  └─ docs/                        ← place PDFs here
├─ scripts/
│  ├─ ingest_docs.py               ← batch ingest CLI
│  └─ healthcheck.py               ← health check endpoint
└─ app/
   ├─ main.py                      ← FastAPI entry point
   ├─ config.py                    ← pydantic-settings
   ├─ schemas.py                   ← all request/response models
   ├─ clients/
   │  └─ local_chroma.py           ← ChromaDB RAG backend
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
