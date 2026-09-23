import asyncio
import json
import uuid

from app.db.chroma_client import ChromaClient
from app.models.schemas import ExtractedMetadata, Relation
from app.pipeline.stage_1_ingestion import IngestionPipeline


class LocalTestMetadataExtractor:
    """Deterministic metadata so this test stays independent of Ollama.

    Without it the real extractor swallows connection errors, returns empty
    metadata, and the assertions below pass on a document that is useless.
    """

    def extract_metadata(self, text):
        return ExtractedMetadata(
            entities=["Entity_A"],
            relations=[Relation(subject="A", predicate="rel", object="B")],
            domain=["Testing"],
        )


class LocalTestEmbeddingFunction:
    """Deterministic local embeddings to keep this test independent of Ollama."""

    def __call__(self, input):
        return [[float((len(text or "") % 7) + i) for i in range(8)] for text in input]

    def name(self) -> str:
        return "local_test_embedding"


def test_pdf_ingestion_into_real_chroma(sample_pdf):
    """Ingest a demo PDF into a real Chroma collection and delete it afterwards.

    """
    coll_name = f"test_integration_{uuid.uuid4().hex[:8]}"
    client = ChromaClient.get_instance()
    collection = client.get_or_create_collection(
        name=coll_name,
        embedding_function=LocalTestEmbeddingFunction(),
    )

    pipeline = IngestionPipeline()
    pipeline.db_collection = collection
    pipeline.metadata_extractor = LocalTestMetadataExtractor()

    try:
        result = asyncio.run(pipeline.process_document(str(sample_pdf)))
        assert result.get("status") == "success"
        contents = collection.get()
        ids = contents.get("ids") or []
        assert ids, "No ids returned from Chroma collection after ingestion"

        stored = (contents.get("metadatas") or [])[0]
        assert json.loads(stored["entities"]) == ["Entity_A"]
        assert json.loads(stored["relations"]) == [["A", "rel", "B"]]
    finally:
        try:
            client.delete_collection(name=coll_name)
        except Exception:
            pass
