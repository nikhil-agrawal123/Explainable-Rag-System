"""
Tests for the query and retrieval endpoints.

    POST /api/v3/query_decomposition/  — Break query into sub-queries
    POST /api/v3/query/                — Multi-query retrieval (RAG)
"""


# =====================================================================
#  Query decomposition
# =====================================================================

class TestQueryDecomposition:
    """Break a complex query into sub-queries."""

    def test_valid_query(self, test_client):
        resp = test_client.post(
            "/api/v3/query_decomposition/",
            params={"query": "What is quantum entanglement and how does it relate to teleportation?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["sub_queries"], list)
        assert len(body["sub_queries"]) >= 1

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


# =====================================================================
#  Multi-query retrieval (RAG)
# =====================================================================

class TestQueryRAG:
    """Multi-query retrieval endpoint."""

    def test_valid_query_returns_chunks(self, test_client):
        resp = test_client.post(
            "/api/v3/query/",
            params={"query": "How do stars form?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["retrieved_chunks"], list)
        assert len(body["retrieved_chunks"]) > 0

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

    def test_chunks_have_expected_fields(self, test_client):
        resp = test_client.post(
            "/api/v3/query/",
            params={"query": "Solar system"},
        )
        chunk = resp.json()["retrieved_chunks"][0]
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "document_id" in chunk
        assert "source" in chunk
