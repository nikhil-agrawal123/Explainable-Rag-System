"""
Tests for the root health-check endpoint.

    GET /
"""


class TestRootEndpoint:
    """GET / — simple liveness probe."""

    def test_root_returns_200(self, test_client):
        resp = test_client.get("/")
        assert resp.status_code == 200

    def test_root_body_has_message(self, test_client):
        data = test_client.get("/").json()
        assert "message" in data
        assert "Running" in data["message"]
