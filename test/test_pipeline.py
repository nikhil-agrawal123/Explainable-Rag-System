"""
Tests for the full pipeline and knowledge graph endpoints.

    POST /api/v3/visualize_graph/  — Build & visualize knowledge graph
    POST /api/v3/full_pipeline/    — End-to-end RAG (query → graph → answer)
"""
from unittest.mock import patch


# =====================================================================
#  Knowledge graph visualization
# =====================================================================

class TestVisualizeGraph:
    """Knowledge graph visualization endpoint."""

    def _graph_patches(self):
        """Context-manager stack that mocks out graph file I/O."""
        return (
            patch(
                "app.utils.visualizer.GraphVisualizer.generate_2d_html",
                return_value="/tmp/test_2d.html",
            ),
            patch(
                "app.utils.visualizer.GraphVisualizer.generate_3d_html",
                return_value="/tmp/test_3d.html",
            ),
            patch(
                "app.utils.visualizer.GraphVisualizer.prune_graph",
                create=True,
                return_value=None,
            ),
        )

    def test_returns_graph_stats(self, test_client):
        p2d, p3d, prune = self._graph_patches()
        with p2d, p3d, prune:
            resp = test_client.post(
                "/api/v3/visualize_graph/",
                params={"query": "Tell me about the solar system"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "stats" in body
        assert "nodes" in body["stats"]
        assert "edges" in body["stats"]

    def test_returns_graph_paths(self, test_client):
        p2d, p3d, prune = self._graph_patches()
        with p2d, p3d, prune:
            resp = test_client.post(
                "/api/v3/visualize_graph/",
                params={"query": "gravity"},
            )
        body = resp.json()
        assert "2d_graph_path" in body
        assert "3d_graph_path" in body

    def test_relational_content_present(self, test_client):
        p2d, p3d, prune = self._graph_patches()
        with p2d, p3d, prune:
            resp = test_client.post(
                "/api/v3/visualize_graph/",
                params={"query": "orbits of planets"},
            )
        body = resp.json()
        assert "relational_content" in body


# =====================================================================
#  Full end-to-end RAG pipeline
# =====================================================================

class TestFullPipeline:
    """End-to-end RAG pipeline (query → answer)."""

    def _viz_patches(self):
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
        assert body["status"] == "success"
        assert "final_answer" in body
        assert len(body["final_answer"]) > 0

    def test_empty_query_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v3/full_pipeline/",
            params={"query": ""},
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
        assert "sub_queries" in body
        assert isinstance(body["sub_queries"], list)

    def test_knowledge_graph_stats_present(self, test_client):
        p2d, p3d = self._viz_patches()
        with p2d, p3d:
            resp = test_client.post(
                "/api/v3/full_pipeline/",
                params={"query": "Neural networks"},
            )
        body = resp.json()
        assert "knowledge_graph_stats" in body
        assert "nodes" in body["knowledge_graph_stats"]
        assert "edges" in body["knowledge_graph_stats"]

    def test_whitespace_only_query(self, test_client):
        resp = test_client.post(
            "/api/v3/full_pipeline/",
            params={"query": "   "},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()
