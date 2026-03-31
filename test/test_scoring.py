"""
Tests for the trust scoring endpoint.

    POST /api/v3/score_trust/  — Calculate trust scores for a generated answer
"""


class TestScoreTrust:
    """Trust-scoring for a generated answer."""

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
        assert body["status"] == "success"
        assert "trust_report" in body
        assert "trust_metrics" in body["trust_report"]

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

    def test_trust_report_structure(self, test_client):
        resp = test_client.post(
            "/api/v3/score_trust/",
            params={
                "query": "Astronomy basics",
                "final_answer": "Stars produce light [TestDoc_CH0].",
            },
        )
        report = resp.json()["trust_report"]
        assert "utilization_rate" in report["trust_metrics"]
        assert "total_citations" in report["trust_metrics"]
        assert "hallucinations" in report["trust_metrics"]
        assert "document_breakdown" in report
        assert "verification_status" in report

    def test_whitespace_answer_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v3/score_trust/",
            params={"query": "Something", "final_answer": "   "},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()
