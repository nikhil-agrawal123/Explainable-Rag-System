"""
Shared fixtures for API endpoint tests.

Provides a patched FastAPI TestClient that mocks out all heavy
dependencies (ChromaDB, Ollama, Whisper, LangSmith tracing) so
tests run fast and without external services.
"""
import pytest
from unittest.mock import patch, MagicMock
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
# Fake pipeline methods — deterministic, zero-cost replacements
# ---------------------------------------------------------------------------

def _fake_decompose(self, query: str):
    return [f"sub-query about {query}", query]


def _fake_retrieve(self, sub_queries, k_per_query=5):
    from app.models.schemas import ChunkRecord, ExtractedMetadata, Relation

    return [
        ChunkRecord(
            chunk_id="TestDoc_CH0",
            document_id="TestDoc",
            text="The sun is a star at the center of the Solar System.",
            source="test.pdf",
            page_number=1,
            metadata=ExtractedMetadata(
                entities=["Sun", "Solar System"],
                relations=[Relation(subject="Sun", predicate="is center of", object="Solar System")],
                domain=["Astronomy"],
            ),
        ),
        ChunkRecord(
            chunk_id="TestDoc_CH1",
            document_id="TestDoc",
            text="Earth orbits the Sun once every 365.25 days.",
            source="test.pdf",
            page_number=2,
            metadata=ExtractedMetadata(
                entities=["Earth", "Sun"],
                relations=[Relation(subject="Earth", predicate="orbits", object="Sun")],
                domain=["Astronomy"],
            ),
        ),
    ]


def _fake_generate(self, query, chunks, graph_summary):
    return f"Generated answer for: {query} [TestDoc_CH0] [TestDoc_CH1]"


def _fake_extract_entities(text):
    return ["Entity_A", "Entity_B"]


def _fake_extract_relations(text):
    return [["Entity_A", "relates_to", "Entity_B"]]


def _fake_transcribe(self, file_path):
    return {"status": "success", "doc_id": "audio_sample", "chunks": 2}


def _fake_ingestion_init(self):
    self.db_collection = FakeChromaClient.get_collection()
    self.chunker = MagicMock()
    self.model = MagicMock()


def _fake_retrieval_init(self):
    self.collection = FakeChromaClient.get_collection()
    self.decomposition_pipeline = MagicMock()
    self.metadata_extractor = MagicMock()


def _fake_decomposition_init(self):
    self.llm = MagicMock()


def _fake_generation_init(self, model_name="qwen2.5:7b"):
    self.llm = MagicMock()
    self.prompt = MagicMock()
    self.chain = MagicMock()


def _fake_trust_scores(self, chunks, answer):
    return {
        "trust_metrics": {
            "utilization_rate": "100.0%",
            "total_citations": 2,
            "hallucinations": 0,
        },
        "document_breakdown": [
            {"document_id": "TestDoc", "citations": 2, "contribution_percent": 100.0}
        ],
        "verification_status": "HIGH",
    }


# ---------------------------------------------------------------------------
# Session-scoped TestClient — built once for the entire test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_client():
    """
    Create a TestClient with all heavy backends mocked out.
    Session-scoped so the app is only built once for the full test run.
    """
    with (
        patch("app.db.chroma_client.ChromaClient", FakeChromaClient),
        patch("app.db.chroma_client.embedding_function", MagicMock()),
        patch("app.db.chroma_client.ollama", MagicMock()),
        patch("app.pipeline.stage_1_ingestion.IngestionPipeline.__init__", _fake_ingestion_init),
        patch("app.pipeline.stage_1_ingestion.IngestionPipeline.transcribe_audio", _fake_transcribe),
        patch("app.pipeline.stage_2_decomposition.QueryDecompositionPipeline.__init__", _fake_decomposition_init),
        patch("app.pipeline.stage_2_decomposition.QueryDecompositionPipeline.decompose_query", _fake_decompose),
        patch("app.pipeline.stage_3_retrieval.MultiQueryRetrievalPipeline.__init__", _fake_retrieval_init),
        patch("app.pipeline.stage_3_retrieval.MultiQueryRetrievalPipeline.retrieve_documents", _fake_retrieve),
        patch("app.pipeline.stage_5_generation.GenerationEngine.__init__", _fake_generation_init),
        patch("app.pipeline.stage_5_generation.GenerationEngine.generate_answer", _fake_generate),
        patch("app.pipeline.stage_6_scoring.TrustScorer.calculate_scores", _fake_trust_scores),
        patch("app.utils.llm.extract_entities", _fake_extract_entities),
        patch("app.utils.llm.extract_relations", _fake_extract_relations),
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
