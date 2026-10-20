# Cross-encoder reranking (BAAI/bge-reranker-v2-m3)
# - Scores (query, chunk) pairs, keeps the most relevant
# - Loaded once at app startup via load_reranker()

import logging
from functools import lru_cache

from langsmith import traceable

from app.core.config import settings
from app.models.schemas import ChunkRecord

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_reranker():
    """Load the cross-encoder once per process; first call downloads the weights."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(settings.RERANKER_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(settings.RERANKER_MODEL).to(device).eval()
    logger.info("Loaded reranker %s on %s.", settings.RERANKER_MODEL, device)
    return tokenizer, model, device


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    import torch

    tokenizer, model, device = load_reranker()
    with torch.no_grad():
        inputs = tokenizer([[query, t] for t in texts], padding=True, truncation=True,
                           return_tensors="pt", max_length=512).to(device)
        return model(**inputs, return_dict=True).logits.view(-1).float().tolist()


@traceable(name="Rerank", run_type="tool")
def rerank(query: str, chunks: list[ChunkRecord], top_n: int = 10) -> list[ChunkRecord]:
    """Score (query, chunk) pairs with the cross-encoder and keep the top_n, best first."""
    if not chunks:
        return []

    for chunk, score in zip(chunks, rerank_scores(query, [c.text for c in chunks])):
        chunk.rerank_score = score

    ranked = sorted(chunks, key=lambda c: c.rerank_score, reverse=True)[:top_n]
    logger.info("Reranked %d chunks, kept %d.", len(chunks), len(ranked))
    return ranked
