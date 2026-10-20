"""Reranker tests — cross-encoder stubbed, runs against every retrieval method."""
import pytest

from app.pipeline import reranker
from test.test_bm25_retrieval import make_pipeline

RETRIEVERS = ["retrieve_documents", "retrieve_documents_bm25", "retrieve_hybrid"]


@pytest.fixture(autouse=True)
def stub_cross_encoder(monkeypatch):
    # Score = number of times "quantum" appears, so the expected order is known.
    monkeypatch.setattr(reranker, "rerank_scores",
                        lambda query, texts: [float(t.lower().count("quantum")) for t in texts])


@pytest.mark.parametrize("method", RETRIEVERS)
def test_rerank_sorts_scores_and_truncates(method):
    chunks = getattr(make_pipeline(), method)(["quantum"], k_per_query=5, user_id="u1")

    ranked = reranker.rerank("quantum", chunks, top_n=2)

    assert ranked[0].chunk_id == "DOC_CH4"  # four "quantum"s
    assert ranked[0].rerank_score == 4.0
    assert len(ranked) == min(2, len(chunks))
    scores = [c.rerank_score for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rerank_handles_no_results():
    assert reranker.rerank("quantum", []) == []
