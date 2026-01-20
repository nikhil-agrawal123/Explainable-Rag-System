
# Explainable RAG System (DataForge Pipeline)

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-111111)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-TBD-lightgrey)](#license)

An **explainable Retrieval-Augmented Generation (RAG)** pipeline designed to move from *“trust the model”* to *“here’s the exact evidence and reasoning trace.”*

This repo follows the pipeline described in **DataForge 2nd Round**: [DataForge_nikhil24380.pdf](DataForge_nikhil24380.pdf).

---

## Why this exists

Most RAG systems answer questions, but do not faithfully show:

- What knowledge was actually used
- Which retrieved sources contributed to the final answer
- Which sources were retrieved but unused (and why)
- How reasoning was structured from evidence

This project’s goal is to output:

1) A natural-language answer
2) A structured explanation in terms of **entities + relationships (triples)** derived directly from retrieved chunks

---

## Pipeline at a glance

```mermaid
flowchart TD
	A[Documents] --> B[Stage 1: Chunking + metadata]
	B --> C[Vector store]
	D[User Query] --> E[Stage 2: Query decomposition]
	E --> F[Stage 3: Multi-query retrieval]
	F --> G[Hybrid retrieval + grader]
	G --> H[Stage 4: Local evidence graph (triples)]
	H --> I[Stage 5: Answer generation + validation]
	I --> J[Stage 6: Document contribution scoring]
	J --> K[Stage 7: User-visible explainable output]
```

---

## Stages (DataForge spec)

### Stage 1 — Data storage and chunking
Chunks are stored with **rich metadata** (topics, parent document id, key entities, etc.).
Embeddings are used for retrieval, while metadata remains stable so you can later explain:

- what got retrieved
- what got used
- what got ignored

### Stage 2 — Query decomposition
Break a complex user question into sub-queries and (optionally) tag them by domain.
This makes retrieval more robust and produces a visible “chain of queries”.

### Stage 3 — Multi-query retrieval (hybrid + graded)
Run retrieval per sub-query, keep results separated, then merge.

- Hybrid retrieval: vector + keyword to reduce false negatives
- Second-stage grader: filters/validates chunks and keeps logs for transparency

### Stage 4 — Local graph creation (evidence-only)
From retrieved chunks, extract:

- nodes: entities
- edges: relations

This produces a local “evidence graph” made only from retrieved text.

### Stage 5 — Answer generation and validation
Generate the answer using only:

- retrieved evidence chunks
- the local graph (nodes/edges)
- sub-queries

Then validate claims against the evidence graph, attach citations (chunk id + document id), and flag unsupported claims.

### Stage 6 — Document scoring and transparency
Quantify contribution per document:

$$
	ext{doc contribution} = \frac{\text{chunks used from doc}}{\text{total chunks retrieved from doc}}
$$

Documents below a threshold (or 0) are marked **retrieved but unused**.

### Stage 7 — User-visible explainable output
The response should include:

- sub-queries (and optionally domains)
- evidence list (document id, chunk id)
- extracted KG triples
- per-document contribution percentages
- unused documents (with explanation)
- the final answer plus a validation breakdown

---

## What’s currently in this repo

- A Python project scaffold with FastAPI and vector DB dependencies.
- Chroma examples:
	- [chroma/persistant.py](chroma/persistant.py) uses `chromadb.PersistentClient`.
	- [chroma/testing.py](chroma/testing.py) shows basic embedding + add/query patterns.

Note: the end-to-end pipeline is described in the PDF and this repository is in an early scaffold phase.

---

## Quickstart (Windows)

### 1) Create environment and install dependencies

This repo includes `uv.lock`. If you use `uv`:

```bash
uv sync
```

If you prefer plain venv + pip:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -e .
```

### 2) Configure environment variables

Create a `.env` file in the repo root. Typical keys you may need:

```ini
OPENAI_API_KEY=...
```

### 3) Run the API

```bash
uvicorn app.main:app --reload
```

---

## Proposed API response shape (recommended)

When the pipeline is wired end-to-end, a good response contract is:

```json
{
	"answer": "...",
	"sub_queries": ["..."],
	"domains_explored": ["..."],
	"evidence": [
		{
			"document_id": "doc-123",
			"chunk_id": "chunk-7",
			"text": "...",
			"score": 0.82
		}
	],
	"triples": [
		{"subject": "...", "predicate": "...", "object": "...", "chunk_id": "chunk-7"}
	],
	"document_contributions": [
		{"document_id": "doc-123", "contribution": 0.5}
	],
	"unused_documents": [
		{"document_id": "doc-999", "reason": "retrieved but no chunks used after grading/validation"}
	],
	"validation": {
		"supported_claims": 5,
		"unsupported_claims": 1,
		"flags": ["Claim 3 not supported by evidence graph"]
	}
}
```

---

## Project structure

```text
.
├─ app/
│  ├─ api/          # Endpoints & dependencies
│  ├─ core/         # Config & environment settings
│  ├─ db/           # Database clients (Chroma, etc.)
│  ├─ models/       # Pydantic schemas
│  ├─ pipeline/     # Logic for Stages 1–7 (Ingestion, Graph, Retrieval)
│  └─ main.py       # FastAPI entry point
├─ data/            # Raw documents
├─ logs/            # Application logs
├─ pyproject.toml
└─ uv.lock
```

---

## Roadmap

- Ingestion: chunking + evidence-aware metadata schema
- Retrieval: multi-query decomposition + hybrid search + grading logs
- Explainability: entity/relation extraction → local graph (triples)
- Generation: answer constrained to evidence + claim validation
- Scoring: document contribution and “unused source” explanations
- UI: optional React dashboard to inspect evidence and reasoning trace

---

## References

See [DataForge_nikhil24380.pdf](DataForge_nikhil24380.pdf) and these starting points:

- https://docs.langchain.com
- https://arxiv.org/abs/2312.10997

---

## License

TBD

