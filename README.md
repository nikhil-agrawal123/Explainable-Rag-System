
# Explainable RAG System (DataForge Pipeline)

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-111111)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Local%20LLM-Ollama-000000)](https://ollama.ai/)
[![LangSmith](https://img.shields.io/badge/Tracing-LangSmith-1C3C3C)](https://smith.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](#license)

An **explainable Retrieval-Augmented Generation (RAG)** pipeline designed to move from *"trust the model"* to *"here's the exact evidence and reasoning trace."*

This repo follows the pipeline described in **DataForge 2nd Round**: [DataForge_nikhil24380.pdf](DataForge_nikhil24380.pdf).

---

## Features

- **Local Embeddings** — Ollama + Qwen3-embedding:8b for high-quality 4096-dim embeddings (no API calls)
- **Domain Classification** — Local LLM (Ollama + Qwen2.5) automatically classifies chunks by knowledge domain
- **Entity & Relation Extraction** — SpaCy transformer model extracts entities and knowledge triples
- **Chunking with Metadata** — LangChain text splitters with rich metadata preservation
- **Vector Store** — ChromaDB persistent storage for embeddings
- **Observability** — Full LangSmith tracing for debugging and transparency
- **Explainable Output** — Evidence graphs, document contributions, and validation traces
- **Fully Local** — No external API dependencies for core functionality

---

## Quickstart (Windows)

### 1) Create environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Install SpaCy model

```bash
python -m spacy download en_core_web_trf
```

### 3) Install Ollama and pull the required models

Download [Ollama](https://ollama.ai/download) and pull the models:

```bash
ollama pull qwen3-embedding:8b
ollama pull qwen2.5:7b
```

### 4) Configure environment variables

Create a `.env` file:

```ini
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=explainable-rag
OLLAMA_HOST=http://localhost:11434
OLLAMA_DOMAIN_MODEL=qwen2.5:7b
OLLAMA_EMBEDDING_MODEL=qwen3-embedding:8b
```

### 5) Run the API

```bash
uvicorn app.main:app --reload
```

Access the API docs at `http://localhost:8000/docs`

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v3/ingest/` | POST | Upload and process PDF documents |
| `/api/v3/full_pipeline/` | POST | Full RAG pipeline (decompose -> retrieve -> graph -> generate) |
| `/api/v3/score_trust/` | POST | Calculate trust scores for a generated answer |
| `/api/v3/query_decomposition/` | POST | Break a query into sub-queries |
| `/api/v3/extract_entities/` | POST | Extract named entities from text |
| `/api/v3/extract_relations/` | POST | Extract subject-predicate-object triples |
| `/api/v3/visualize_graph/` | POST | Generate interactive knowledge graph visualizations |

---

## Project Structure

```
app/
  api/          # Endpoints & dependencies
  core/         # Config & environment settings
  db/           # ChromaDB client
  models/       # Pydantic schemas
  pipeline/     # Stages 1-6 logic
  utils/        # LLM & visualization utilities
  main.py       # FastAPI entry point
data/
  chroma_storage/  # Persistent vector DB
  uploads/         # Uploaded documents
test/            # Test files
requirements.txt
```

---

## Troubleshooting

### Ollama Connection Errors

```bash
# Check if Ollama is running
ollama list

# Verify models are downloaded
ollama pull qwen3-embedding:8b
ollama pull qwen2.5:7b
```

### SpaCy Model Errors

```bash
python -m spacy download en_core_web_trf
```

---

## References

- [DataForge_nikhil24380.pdf](DataForge_nikhil24380.pdf)
- https://docs.langchain.com
- https://arxiv.org/abs/2312.10997

---

## License

MIT
