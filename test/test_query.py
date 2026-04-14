"""
Tests for the query and retrieval endpoints.

    POST /api/v3/query_decomposition/  — Break query into sub-queries
    POST /api/v3/query/                — Multi-query retrieval (RAG)

These tests exercise the REAL decomposition parsing logic in
stage_2_decomposition.py. The FakeLLM returns numbered lines,
and the real parsing code (line splitting, number detection) is tested.
"""


# =====================================================================
#  Query decomposition
# =====================================================================

class TestQueryDecomposition:
    """Break a complex query into sub-queries (real parsing, fake LLM)."""

    def test_valid_query_returns_sub_queries(self, test_client):
        """The real decompose_query() should parse FakeLLM's numbered list."""
        resp = test_client.post(
            "/api/v3/query_decomposition/",
            params={"query": "What is quantum entanglement and how does it relate to teleportation?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["sub_queries"], list)
        # FakeLLM returns 3 numbered lines + the original query is appended
        # So we expect at least 4 sub-queries
        assert len(body["sub_queries"]) >= 4

    def test_original_query_included_in_sub_queries(self, test_client):
        """The original query should always be appended to sub-queries."""
        q = "Explain photosynthesis."
        resp = test_client.post(
            "/api/v3/query_decomposition/",
            params={"query": q},
        )
        body = resp.json()
        assert q in body["sub_queries"]

    def test_empty_query_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v3/query_decomposition/",
            params={"query": ""},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_original_query_echoed(self, test_client):
        q = "Explain photosynthesis."
        resp = test_client.post(
            "/api/v3/query_decomposition/",
            params={"query": q},
        )
        assert resp.json()["original_query"] == q

    def test_sub_queries_are_strings(self, test_client):
        """All sub-queries should be plain strings."""
        resp = test_client.post(
            "/api/v3/query_decomposition/",
            params={"query": "How do computers work?"},
        )
        for sq in resp.json()["sub_queries"]:
            assert isinstance(sq, str)
            assert len(sq.strip()) > 0


# =====================================================================
#  Multi-query retrieval (RAG)
# =====================================================================

class TestQueryRAG:
    """Multi-query retrieval endpoint — exercises real decomposition + retrieval."""

    def test_valid_query_returns_success(self, test_client):
        resp = test_client.post(
            "/api/v3/query/",
            params={"query": "How do stars form?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        # May return success with empty chunks (FakeCollection starts empty)
        # or success with chunks if data was seeded
        assert body.get("status") == "success" or "error" not in body

    def test_empty_query_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v3/query/",
            params={"query": "  "},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_sub_queries_included(self, test_client):
        resp = test_client.post(
            "/api/v3/query/",
            params={"query": "Climate change effects"},
        )
        body = resp.json()
        assert "sub_queries" in body
        assert isinstance(body["sub_queries"], list)

    def test_retrieved_chunks_is_list(self, test_client):
        resp = test_client.post(
            "/api/v3/query/",
            params={"query": "Solar system"},
        )
        body = resp.json()
        assert "retrieved_chunks" in body
        assert isinstance(body["retrieved_chunks"], list)
