"""
Shared fixtures for API endpoint tests.

Provides a patched FastAPI TestClient that mocks out ONLY external
boundaries (Ollama LLM, ChromaDB, Whisper) so tests exercise real
application logic while remaining fast and offline.
"""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fake / stub objects — in-memory replacements for external services
# ---------------------------------------------------------------------------

class FakeCollection:
    """In-memory stand-in for a ChromaDB collection."""

    def __init__(self):
        self._store: dict = {}

    def add(self, ids, documents, metadatas):
        for i, cid in enumerate(ids):
            self._store[cid] = (documents[i], metadatas[i])

    def upsert(self, ids, documents, metadatas):
        """Same as add but overwrites existing entries (mirrors real ChromaDB)."""
        for i, cid in enumerate(ids):
            self._store[cid] = (documents[i], metadatas[i])

    def get(self, where=None, limit=None, **kwargs):
        """Minimal get() — returns empty results by default (no duplicates)."""
        return {"ids": [], "documents": [], "metadatas": []}

    def query(self, query_texts, n_results, include=None):
        all_ids = list(self._store.keys())[:n_results]
        docs = [self._store[cid][0] for cid in all_ids]
        metas = [self._store[cid][1] for cid in all_ids]
        return {
            "ids": [all_ids] * len(query_texts),
            "documents": [docs] * len(query_texts),
            "metadatas": [metas] * len(query_texts),
            "distances": [[0.1] * len(all_ids)] * len(query_texts),
        }


class FakeChromaClient:
    """Singleton-compatible replacement for ChromaClient."""

    _collection = FakeCollection()

    @classmethod
    def get_instance(cls):
        return MagicMock()

    @classmethod
    def get_collection(cls, name=None):
        return cls._collection


# ---------------------------------------------------------------------------
# Fake LLM — returns deterministic structured content
# ---------------------------------------------------------------------------

class FakeAIMessage:
    """Mimics langchain AIMessage with a .content attribute."""
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """
    A deterministic fake LLM that returns structured responses based on
    the prompt content. This allows real parsing logic to be tested.
    """
    def invoke(self, messages, **kwargs):
        # Inspect the prompt to determine what kind of call this is
        prompt_text = ""
        for m in messages:
            if hasattr(m, "content"):
                prompt_text += m.content + "\n"
            elif isinstance(m, tuple):
                prompt_text += str(m[1]) + "\n"

        prompt_lower = prompt_text.lower()

        # Query decomposition
        if "decomposition" in prompt_lower or "sub-quer" in prompt_lower:
            return FakeAIMessage(
                "1. What are the fundamental concepts?\n"
                "2. How do they relate to each other?\n"
                "3. What are the practical applications?"
            )

        # Domain classification
        if "domain classification" in prompt_lower:
            return FakeAIMessage("Computer Science")

        # Entity extraction
        if "extract entities" in prompt_lower or "entity extraction" in prompt_lower:
            return FakeAIMessage('[\"Entity_A\", \"Entity_B\", \"Entity_C\"]')

        # Relation extraction
        if "extract relations" in prompt_lower or "relation extraction" in prompt_lower or "knowledge extraction" in prompt_lower:
            return FakeAIMessage(
                '[[\"Entity_A\", \"relates_to\", \"Entity_B\"], '
                '[\"Entity_B\", \"is_part_of\", \"Entity_C\"]]'
            )

        # Generation (answer)
        if "dataforge" in prompt_lower and "evidence-based" in prompt_lower:
            return FakeAIMessage(
                "Based on the evidence, the answer involves key concepts "
                "[TestDoc_CH0] and supported relationships [TestDoc_CH1]."
            )

        # Default
        return FakeAIMessage("Default LLM response.")


_fake_llm_instance = FakeLLM()


def _get_fake_llm():
    return _fake_llm_instance


# ---------------------------------------------------------------------------
# Fake transcription for audio tests
# ---------------------------------------------------------------------------

def _fake_transcribe(self, file_path):
    return {"status": "success", "doc_id": "audio_sample", "chunks": 2}


# ---------------------------------------------------------------------------
# Session-scoped TestClient — built once for the entire test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_client():
    """
    Create a TestClient with external boundaries mocked.
    
    What IS mocked (external services):
      - ChromaDB client/collection → FakeCollection (in-memory)
      - Ollama LLM → FakeLLM (deterministic responses)
      - Ollama embeddings → MagicMock
      - Whisper transcription → fake return value
    
    What is NOT mocked (real logic that gets tested):
      - Query decomposition parsing (stage_2)
      - Entity/relation extraction JSON parsing (llm.py)
      - Knowledge graph construction (stage_4)
      - Trust scoring & citation extraction (stage_6)
      - Generation prompt assembly (stage_5)
      - All FastAPI routing and validation
    """
    with (
        # External boundary: ChromaDB
        patch("app.db.chroma_client.ChromaClient", FakeChromaClient),
        patch("app.db.chroma_client.embedding_function", MagicMock()),
        patch("app.db.chroma_client.ollama", MagicMock()),
        # External boundary: Ollama LLM — use FakeLLM via get_llm()
        patch("app.utils.llm.get_llm", _get_fake_llm),
        patch("app.utils.llm._llm", _fake_llm_instance),
        # External boundary: Whisper audio model
        patch(
            "app.pipeline.stage_1_ingestion.IngestionPipeline.transcribe_audio",
            _fake_transcribe,
        ),
        # Patch ChromaClient in modules that import it directly
        patch("app.pipeline.stage_1_ingestion.ChromaClient", FakeChromaClient),
        patch("app.pipeline.stage_3_retrieval.ChromaClient", FakeChromaClient),
    ):
        from app.main import app

        client = TestClient(app)
        yield client


# ---------------------------------------------------------------------------
# File-creation fixtures for upload tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_pdf(tmp_path):
    """Create a minimal valid PDF for upload tests."""
    pdf_bytes = (
        b"%PDF-1.0\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\n"
        b"xref\n0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n210\n%%EOF"
    )
    path = tmp_path / "test_upload.pdf"
    path.write_bytes(pdf_bytes)
    return path
