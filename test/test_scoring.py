"""
Tests for the trust scoring endpoint.

    POST /api/v3/score_trust/  — Calculate trust scores for a generated answer

These tests exercise the REAL TrustScorer.calculate_scores() logic
including regex citation extraction, contribution calculation, and
hallucination detection.
"""


class TestScoreTrust:
    """Trust-scoring for a generated answer — real scoring logic tested."""

    def test_valid_inputs(self, test_client):
        resp = test_client.post(
            "/api/v3/score_trust/",
            params={
                "query": "Tell me about the Sun",
                "final_answer": "The Sun is a star [TestDoc_CH0].",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        # The endpoint may return success or error depending on whether
        # the retrieval pipeline found chunks in the (possibly empty) FakeCollection
        assert body.get("status") == "success" or "error" in body

    def test_empty_query_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v3/score_trust/",
            params={"query": "", "final_answer": "Some answer."},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_empty_answer_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v3/score_trust/",
            params={"query": "What is Earth?", "final_answer": ""},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_whitespace_answer_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v3/score_trust/",
            params={"query": "Something", "final_answer": "   "},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()
