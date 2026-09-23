"""BM25 + hybrid retrieval tests — no Chroma, no Ollama, no LLM."""
import json

import pytest

from app.pipeline import stage_3_retrieval
from app.pipeline.stage_3_retrieval import MultiQueryRetrievalPipeline

CORPUS = {
    "DOC_CH0": "Quantum entanglement links two particles across any distance.",
    "DOC_CH1": "Bayesian inference updates a prior belief with observed evidence.",
    "DOC_CH2": "Entanglement is central to quantum teleportation experiments.",
    "DOC_CH3": "The mitochondria produce energy inside the cell.",
    "DOC_CH4": "Quantum quantum quantum computing relies on quantum gates.",
}


class StubCollection:
    """Minimal stand-in for a Chroma collection holding one user's chunks."""

    def __init__(self, corpus=CORPUS, user_id="u1"):
        self.corpus = corpus
        self.user_id = user_id
        self.last_where = "unset"
        self.calls = []

    def _meta(self):
        return {"document_id": "DOC", "source": "doc.pdf", "page": 1, "user_id": self.user_id,
                "entities": json.dumps(["Entity_A"]), "relations": json.dumps([["A", "rel", "B"]]),
                "domain": json.dumps(["Physics"])}

    def get(self, where=None, include=None, ids=None, **kwargs):
        self.calls.append({"where": where, "include": include, "ids": ids})
        self.last_where = where
        if ids is not None:
            present = [i for i in ids if i in self.corpus]
            return {"ids": present,
                    "documents": [self.corpus[i] for i in present],
                    "metadatas": [self._meta() for _ in present]}
        if where and where.get("user_id") != self.user_id:
            return {"ids": [], "documents": [], "metadatas": []}
        if include == []:
            return {"ids": list(self.corpus), "documents": None, "metadatas": None}
        return {
            "ids": list(self.corpus),
            "documents": list(self.corpus.values()),
            "metadatas": [self._meta() for _ in self.corpus],
        }

    def query(self, query_texts, n_results, where=None, include=None):
        ids = list(self.corpus)[:n_results]
        return {
            "ids": [ids] * len(query_texts),
            "documents": [[self.corpus[i] for i in ids]] * len(query_texts),
            "metadatas": [[self._meta() for _ in ids]] * len(query_texts),
            "distances": [[0.1] * len(ids)] * len(query_texts),
        }


@pytest.fixture(autouse=True)
def clear_index_cache():
    """The BM25 index cache is module-level, so tests must not leak into each other."""
    stage_3_retrieval._BM25_INDEX_CACHE.clear()
    yield
    stage_3_retrieval._BM25_INDEX_CACHE.clear()


def make_pipeline(collection=None):
    """Build the pipeline without touching ChromaClient."""
    pipeline = MultiQueryRetrievalPipeline.__new__(MultiQueryRetrievalPipeline)
    pipeline.collection = collection or StubCollection()
    return pipeline


def test_bm25_returns_only_documents_sharing_query_terms():
    chunks = make_pipeline().retrieve_documents_bm25(["quantum entanglement"], k_per_query=2, user_id="u1")

    assert {c.chunk_id for c in chunks} == {"DOC_CH0", "DOC_CH2"}  # the two entanglement chunks
    assert chunks[0].source == "doc.pdf"
    # metadata comes from what ingestion stored, not from a fresh LLM call
    assert chunks[0].metadata.entities == ["Entity_A"]
    assert chunks[0].metadata.domain == ["Physics"]
    assert chunks[0].metadata.relations[0].predicate == "rel"


def test_bm25_ranks_by_term_frequency():
    chunks = make_pipeline().retrieve_documents_bm25(["quantum"], k_per_query=1, user_id="u1")

    assert [c.chunk_id for c in chunks] == ["DOC_CH4"]  # says "quantum" four times


def test_bm25_skips_documents_with_no_query_terms():
    chunks = make_pipeline().retrieve_documents_bm25(["mitochondria"], k_per_query=4, user_id="u1")

    assert [c.chunk_id for c in chunks] == ["DOC_CH3"]


def test_bm25_deduplicates_across_sub_queries():
    chunks = make_pipeline().retrieve_documents_bm25(
        ["quantum entanglement", "entanglement teleportation", "bayesian prior"],
        k_per_query=2,
        user_id="u1",
    )

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert set(ids) == {"DOC_CH0", "DOC_CH2", "DOC_CH1"}


def test_bm25_filters_by_user_and_handles_empty_corpus():
    collection = StubCollection()
    chunks = make_pipeline(collection).retrieve_documents_bm25(["quantum"], k_per_query=5, user_id="other")

    assert chunks == []
    assert collection.last_where == {"user_id": "other"}


def test_hybrid_merges_without_duplicating_chunks():
    pipeline = make_pipeline()
    vector_ids = {c.chunk_id for c in pipeline.retrieve_documents(["quantum"], k_per_query=3, user_id="u1")}
    bm25_ids = {c.chunk_id for c in pipeline.retrieve_documents_bm25(["quantum"], k_per_query=3, user_id="u1")}

    hybrid = pipeline.retrieve_hybrid(["quantum"], k_per_query=3, user_id="u1")
    hybrid_ids = [c.chunk_id for c in hybrid]

    assert vector_ids & bm25_ids, "fixture should overlap so dedup is actually exercised"
    assert len(hybrid_ids) == len(set(hybrid_ids))
    assert set(hybrid_ids) == vector_ids | bm25_ids


def test_bm25_index_is_reused_until_the_corpus_changes():
    collection = StubCollection()
    pipeline = make_pipeline(collection)

    pipeline.retrieve_documents_bm25(["quantum"], k_per_query=1, user_id="u1")
    builds = [c for c in collection.calls if c["include"] == ["documents"]]
    assert len(builds) == 1

    pipeline.retrieve_documents_bm25(["bayesian"], k_per_query=1, user_id="u1")
    builds = [c for c in collection.calls if c["include"] == ["documents"]]
    assert len(builds) == 1, "second query must reuse the cached index"

    collection.corpus = dict(collection.corpus, DOC_CH5="A newly ingested bayesian chunk.")
    chunks = pipeline.retrieve_documents_bm25(["bayesian"], k_per_query=5, user_id="u1")
    builds = [c for c in collection.calls if c["include"] == ["documents"]]
    assert len(builds) == 2, "ingesting a document must invalidate the index"
    assert "DOC_CH5" in {c.chunk_id for c in chunks}
