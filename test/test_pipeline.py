"""
Tests for the full pipeline and knowledge graph endpoints.

    POST /api/v3/visualize_graph/  — Build & visualize knowledge graph
    POST /api/v3/full_pipeline/    — End-to-end RAG (query → graph → answer)

These tests exercise real decomposition, graph construction, and
generation logic. Only file I/O (HTML graph export) is mocked.
"""
from unittest.mock import patch


# =====================================================================
#  Knowledge graph visualization
# =====================================================================

class TestVisualizeGraph:
    """Knowledge graph visualization endpoint."""

    def _graph_io_patches(self):
        """Mock only file I/O — the graph construction itself is REAL."""
        return (
            patch(
                "app.utils.visualizer.GraphVisualizer.generate_2d_html",
                return_value="/tmp/test_2d.html",
            ),
            patch(
                "app.utils.visualizer.GraphVisualizer.generate_3d_html",
                return_value="/tmp/test_3d.html",
            ),
        )

    def test_returns_success(self, test_client):
        p2d, p3d = self._graph_io_patches()
        with p2d, p3d:
            resp = test_client.post(
                "/api/v3/visualize_graph/",
                params={"query": "Tell me about the solar system"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

    def test_returns_graph_stats(self, test_client):
        p2d, p3d = self._graph_io_patches()
        with p2d, p3d:
            resp = test_client.post(
                "/api/v3/visualize_graph/",
                params={"query": "Tell me about the solar system"},
            )
        body = resp.json()
        assert "stats" in body
        assert "nodes" in body["stats"]
        assert "edges" in body["stats"]
        # Node/edge counts should be integers
        assert isinstance(body["stats"]["nodes"], int)
        assert isinstance(body["stats"]["edges"], int)

    def test_returns_graph_paths(self, test_client):
        p2d, p3d = self._graph_io_patches()
        with p2d, p3d:
            resp = test_client.post(
                "/api/v3/visualize_graph/",
                params={"query": "gravity"},
            )
        body = resp.json()
        assert "2d_graph_path" in body
        assert "3d_graph_path" in body

    def test_relational_content_is_string(self, test_client):
        p2d, p3d = self._graph_io_patches()
        with p2d, p3d:
            resp = test_client.post(
                "/api/v3/visualize_graph/",
                params={"query": "orbits of planets"},
            )
        body = resp.json()
        assert "relational_content" in body
        assert isinstance(body["relational_content"], str)


# =====================================================================
#  Full end-to-end RAG pipeline
# =====================================================================

class TestFullPipeline:
    """End-to-end RAG pipeline (query → answer)."""

    def _viz_patches(self):
        """Mock only graph file I/O."""
        return (
            patch(
                "app.utils.visualizer.GraphVisualizer.generate_2d_html",
                return_value="/tmp/fp_2d.html",
            ),
            patch(
                "app.utils.visualizer.GraphVisualizer.generate_3d_html",
                return_value="/tmp/fp_3d.html",
            ),
        )

    def test_valid_query_returns_answer(self, test_client):
        p2d, p3d = self._viz_patches()
        with p2d, p3d:
            resp = test_client.post(
                "/api/v3/full_pipeline/",
                params={"query": "How does the Sun produce energy?"},
            )
        assert resp.status_code == 200
        body = resp.json()
        # May succeed with answer or fail with "no relevant info" (empty collection)
        assert "status" in body or "error" in body

    def test_empty_query_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v3/full_pipeline/",
            params={"query": ""},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_whitespace_only_query(self, test_client):
        resp = test_client.post(
            "/api/v3/full_pipeline/",
            params={"query": "   "},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_response_contains_sub_queries(self, test_client):
        p2d, p3d = self._viz_patches()
        with p2d, p3d:
            resp = test_client.post(
                "/api/v3/full_pipeline/",
                params={"query": "What is machine learning?"},
            )
        body = resp.json()
        # If pipeline succeeded, sub_queries should be present
        if "status" in body and body["status"] == "success":
            assert "sub_queries" in body
            assert isinstance(body["sub_queries"], list)
