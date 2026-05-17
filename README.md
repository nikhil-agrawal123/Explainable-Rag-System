# Explainable RAG System (DataForge Pipeline)

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-111111)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Local%20LLM-Ollama-000000)](https://ollama.ai/)
[![LangSmith](https://img.shields.io/badge/Tracing-LangSmith-1C3C3C)](https://smith.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license)

An **explainable Retrieval-Augmented Generation (RAG)** pipeline designed to move from _"trust the model"_ to _"here's the exact evidence and reasoning trace."_

---

## Features

- **Local Embeddings** — Ollama + Qwen3-embedding for high-quality embeddings (no external API calls)
- **Domain Classification** — Local LLM automatically classifies chunks by knowledge domain
- **Entity & Relation Extraction** — Local LLM extracts entities and knowledge triples
- **Query Decomposition** — Complex queries broken into sub-queries for targeted retrieval
- **Knowledge Graph** — Builds an evidence graph from retrieved chunks with provenance tracking
- **Trust Scoring** — Citation-level analysis quantifying document contribution and hallucination risk
- **Vector Store** — ChromaDB persistent storage for embeddings
- **Full Observability** — LangSmith tracing for every pipeline stage
- **Per-User Isolation** — Uploads, vector data, and visualizations scoped to each authenticated user

---

## Quickstart

### 1) Prerequisites

- Python 3.12+
- [Ollama](https://ollama.ai/download) running locally with required models

```bash
ollama pull qwen3-embedding:8b
ollama pull qwen3.5:9b
```

### 2) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

pip install -U pip
pip install -r requirements.txt
```

### 3) Configure environment

```bash
cp .env.example .env
# Edit .env with your Supabase credentials, JWT secret, and Ollama host
```

### 4) Run the API

```bash
uvicorn app.main:app --reload
```

Access the interactive docs at `http://localhost:8000/docs`

---

## Docker

```bash
docker build -t dataforge .
docker run -p 8000:8000 --env-file .env dataforge
```

> The image bundles the app only. Ollama must run separately (locally or as a linked container). Persistent data (ChromaDB, uploads, viz) is written to ephemeral container storage — mount a volume or use a cloud vector DB for production.

---

## Authentication

DataForge uses **JWT tokens** (HS256) issued after Supabase credential verification.

| Step | Description |
|------|-------------|
| **Signup** | `POST /api/v3/signup` — creates user in Supabase |
| **Login** | `POST /api/v3/login/` — returns `access_token` (JWT) |
| **Authenticate** | Send `Authorization: Bearer <token>` on protected routes |
| **User info** | `GET /api/v3/me/` — returns the authenticated user's email |

Tokens expire after `JWT_EXPIRY_MINUTES` (default 60). The client is responsible for storing the token (e.g., `localStorage`) and discarding it on logout.

---

## API Endpoints

### Auth
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v3/signup` | Create a new account |
| POST | `/api/v3/login/` | Log in and receive a JWT |
| POST | `/api/v3/logout/` | Sign out (client discards token) |
| GET | `/api/v3/me/` | Return authenticated user info |

### Ingestion
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v3/ingest_pdf/` | Upload and process PDF documents |
| POST | `/api/v3/ingest_audio/` | Transcribe and process audio files |

### LLM Tools
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v3/extract_entities/` | Extract named entities from text |
| POST | `/api/v3/extract_relations/` | Extract subject-predicate-object triples |
| POST | `/api/v3/query_decomposition/` | Break a query into sub-queries |

### RAG Pipeline
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v3/query/` | Decompose + retrieve relevant chunks |
| POST | `/api/v3/visualize_graph/` | Build and visualize knowledge graph |
| POST | `/api/v3/full_pipeline/` | Full pipeline: decompose → retrieve → graph → generate |
| POST | `/api/v3/score_trust/` | Calculate trust scores for a generated answer |

### Health
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check (server is running) |
| GET | `/health/ready` | Readiness check (ChromaDB, Ollama, directories) |

---

## Project Structure

```
.
├── app/
│   ├── api/
│   │   ├── auth.py              # JWT creation & validation
│   │   ├── endpoints.py         # Router composition
│   │   └── routes/
│   │       ├── auth.py          # Signup, login, logout, me
│   │       ├── ingestion.py     # PDF & audio upload
│   │       ├── llm_tools.py     # Entity/relation extraction
│   │       └── rag.py           # Query, graph, generation, scoring
│   ├── core/
│   │   └── config.py            # Settings from environment
│   ├── db/
│   │   └── chroma_client.py     # ChromaDB singleton + embedding function
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── pipeline/
│   │   ├── stage_1_ingestion.py    # PDF loading & chunking
│   │   ├── stage_2_decomposition.py# Query decomposition
│   │   ├── stage_3_retrieval.py    # Multi-query retrieval
│   │   ├── stage_4_local_graph.py  # Knowledge graph construction
│   │   ├── stage_5_generation.py   # Answer generation
│   │   ├── stage_6_scoring.py      # Trust scoring
│   │   ├── chuncking.py            # Text splitting
│   │   └── metadata.py             # Entity/relation/domain extraction
│   ├── utils/
│   │   ├── llm.py               # LLM helpers (domain, entities, relations)
│   │   └── visualizer.py        # PyVis & Plotly graph HTML output
│   └── main.py                  # FastAPI entry point
├── data/
│   ├── chroma_storage/          # Persistent vector DB (per-user scoped)
│   ├── uploads/                 # Uploaded files (per-user subdirectories)
│   └── viz/                     # Generated graph HTMLs (per-user subdirectories)
├── dockerfile
├── .dockerignore
├── .env.example
├── requirements.txt
└── test/
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SUPABASE_URL` | Yes | — | Supabase project URL |
| `SUPABASE_KEY` | Yes | — | Supabase anon/public key |
| `JWT_SECRET` | Yes | — | HS256 signing key (generate with `secrets.token_hex(32)`) |
| `OLLAMA_HOST` | Yes | — | Ollama server URL |
| `OLLAMA_DOMAIN_MODEL` | No | `qwen3.5:4b` | LLM for generation, decomposition, extraction |
| `OLLAMA_EMBEDDING_MODEL` | No | `qwen3-embedding:8b` | Model for vector embeddings |
| `JWT_EXPIRY_MINUTES` | No | `60` | Token lifetime in minutes |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `LANGSMITH_API_KEY` | No | — | LangSmith tracing key (optional) |

---

## Troubleshooting

### Ollama Connection Errors
```bash
ollama list                    # Check if Ollama is running
ollama pull qwen3-embedding:8b # Verify models are downloaded
ollama pull qwen3.5:9b
```

### Token / Auth Issues
```bash
# Get a new token
curl -X POST http://localhost:8000/api/v3/login/ \
  -d "email=user@example.com&password=yourpass"

# Use it on protected routes
curl -X POST http://localhost:8000/api/v3/query/ \
  -H "Authorization: Bearer <token>" \
  -d "query=How do stars form?"
```

---

## License

MIT
