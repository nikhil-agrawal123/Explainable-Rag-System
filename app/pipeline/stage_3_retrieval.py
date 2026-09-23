# Stage 3: Multi-Query Retrieval
# - Interact with ChromaDB
# - Hybrid Search (Vector + Keyword)
# - Second-stage grader

import logging
import math
import re
from collections import Counter
from typing import NamedTuple, Optional

from app.db.chroma_client import ChromaClient   
from app.pipeline.stage_2_decomposition import QueryDecompositionPipeline
from langsmith import traceable
from app.models.schemas import ChunkRecord, ExtractedMetadata

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")

BM25_K1 = 1.5
BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class _Bm25Index(NamedTuple):
    """Precomputed BM25 statistics for one user's corpus."""
    fingerprint: int
    ids: list[str]
    freqs: list[Counter]
    doc_lens: list[int]
    avgdl: float
    idf: dict[str, float]

_BM25_INDEX_CACHE: dict[str, _Bm25Index] = {}


def _build_bm25_index(ids: list[str], docs: list[str]) -> _Bm25Index:
    freqs = [Counter(_tokenize(doc)) for doc in docs]
    doc_lens = [sum(f.values()) for f in freqs]
    n = len(freqs)

    df = Counter()
    for f in freqs:
        df.update(f.keys())

    return _Bm25Index(
        fingerprint=hash(tuple(ids)),
        ids=ids,
        freqs=freqs,
        doc_lens=doc_lens,
        avgdl=(sum(doc_lens) / n) or 1.0,
        idf={term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()},
    )


class MultiQueryRetrievalPipeline:
    def __init__(self):
        self.collection = ChromaClient.get_collection()
        self.decomposition_pipeline = QueryDecompositionPipeline()

    @traceable(name="Multi-Query Retrieval", run_type="tool")
    def retrieve_documents(self, sub_queries: list[str], k_per_query: int, user_id: str = "") -> list[ChunkRecord]:
        where_filter = {"user_id": user_id} if user_id else None

        results = self.collection.query(
            query_texts=sub_queries,
            n_results=k_per_query,
            where=where_filter,
            include=["metadatas", "documents", "distances"]
        )

        unique_chunks = {}
        for q_index, doc_list in enumerate(results["documents"]):
            for i, doc_text in enumerate(doc_list):
                chunk_id = results["ids"][q_index][i]
                raw_meta = results["metadatas"][q_index][i]
                
                if chunk_id in unique_chunks:
                    continue

                unique_chunks[chunk_id] = self._build_record(chunk_id, doc_text, raw_meta)

        unique_list = list(unique_chunks.values())
        logger.info("Retrieved %d unique chunks from %d queries.", len(unique_list), len(sub_queries))
        return unique_list

    def _build_record(self, chunk_id: str, doc_text: str, raw_meta: dict) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=chunk_id,
            document_id=raw_meta.get("document_id", "unknown"),
            text=doc_text,
            source=raw_meta.get("source", "unknown"),
            page_number=raw_meta.get("page", 0),
            user_id=raw_meta.get("user_id", ""),
            start_time=raw_meta.get("start_time"),  # Audio timestamps
            end_time=raw_meta.get("end_time"),      # Audio timestamps
            metadata=ExtractedMetadata.from_persistence_payload(raw_meta),
        )

    def _get_bm25_index(self, user_id: str) -> Optional[_Bm25Index]:
        """Return a BM25 index for this user, rebuilding only when their chunk set changed."""
        where_filter = {"user_id": user_id} if user_id else None

        current_ids = self.collection.get(where=where_filter, include=[]).get("ids") or []
        if not current_ids:
            return None

        cached = _BM25_INDEX_CACHE.get(user_id)
        if cached is not None and cached.fingerprint == hash(tuple(current_ids)):
            return cached

        corpus = self.collection.get(where=where_filter, include=["documents"])
        index = _build_bm25_index(corpus.get("ids") or [], corpus.get("documents") or [])
        _BM25_INDEX_CACHE[user_id] = index
        logger.info("Built BM25 index for user %s over %d chunks.", user_id, len(index.ids))
        return index

    @traceable(name="Multi-Query BM25 Retrieval", run_type="tool")
    def retrieve_documents_bm25(self, sub_queries: list[str], k_per_query: int, user_id: str = "") -> list[ChunkRecord]:
        """Keyword (BM25-Okapi) retrieval over the user's chunks stored in Chroma."""
        index = self._get_bm25_index(user_id)
        if index is None:
            logger.info("BM25: no chunks stored for user %s.", user_id)
            return []

        hit_ids = []
        for query in sub_queries:
            terms = _tokenize(query)
            scored = []
            for i, f in enumerate(index.freqs):
                score = 0.0
                for term in terms:
                    tf = f.get(term, 0)
                    if not tf:
                        continue
                    norm = 1 - BM25_B + BM25_B * index.doc_lens[i] / index.avgdl
                    score += index.idf[term] * tf * (BM25_K1 + 1) / (tf + BM25_K1 * norm)
                if score > 0:
                    scored.append((score, i))

            scored.sort(key=lambda s: s[0], reverse=True)
            for _score, i in scored[:k_per_query]:
                chunk_id = index.ids[i]
                if chunk_id not in hit_ids:
                    hit_ids.append(chunk_id)

        if not hit_ids:
            return []

        hits = self.collection.get(ids=hit_ids, include=["documents", "metadatas"])
        by_id = dict(zip(hits.get("ids") or [], zip(hits.get("documents") or [], hits.get("metadatas") or [])))

        records = [self._build_record(cid, by_id[cid][0], by_id[cid][1] or {}) for cid in hit_ids if cid in by_id]
        logger.info("Retrieved %d unique chunks from %d queries using BM25.", len(records), len(sub_queries))
        return records

    @traceable(name="Hybrid Retrieval", run_type="tool")
    def retrieve_hybrid(self, sub_queries: list[str], k_per_query: int, user_id: str = "") -> list[ChunkRecord]:
        """Vector + BM25 results, deduplicated by chunk_id (vector hits win ties)."""
        merged = {}
        for record in (
            self.retrieve_documents(sub_queries, k_per_query, user_id)
            + self.retrieve_documents_bm25(sub_queries, k_per_query, user_id)
        ):
            merged.setdefault(record.chunk_id, record)

        logger.info("Hybrid retrieval returned %d unique chunks.", len(merged))
        return list(merged.values())
