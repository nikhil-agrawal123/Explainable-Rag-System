"""Ingestion must not store a document whose metadata extraction failed."""
import json

import pytest

from app.models.schemas import ChunkRecord, ExtractedMetadata, Relation
from app.pipeline.stage_1_ingestion import _reject_if_extraction_failed


def payload(metadata):
    return ChunkRecord(chunk_id="D_CH0", document_id="D", text="t", source="d.pdf",
                       metadata=metadata).to_persistence_payload()


def test_rejects_when_every_chunk_came_back_empty():
    metas = [payload(ExtractedMetadata()) for _ in range(3)]

    with pytest.raises(ValueError, match="Ollama is reachable"):
        _reject_if_extraction_failed(metas, "D")


def test_accepts_when_any_chunk_has_entities():
    metas = [payload(ExtractedMetadata()), payload(ExtractedMetadata(entities=["Bayes"]))]

    _reject_if_extraction_failed(metas, "D")  # must not raise


def test_accepts_when_only_relations_were_found():
    rel = Relation(subject="A", predicate="rel", object="B")
    metas = [payload(ExtractedMetadata(relations=[rel]))]

    _reject_if_extraction_failed(metas, "D")


def test_domain_alone_is_not_enough():
    """domain defaults to ["General"] even on total failure, so it can't signal success."""
    metas = [payload(ExtractedMetadata(domain=["Physics"]))]

    assert json.loads(metas[0]["entities"]) == []
    with pytest.raises(ValueError):
        _reject_if_extraction_failed(metas, "D")
