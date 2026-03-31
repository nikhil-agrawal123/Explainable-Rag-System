"""
Tests for the NLP extraction endpoints.

    POST /api/v3/extract_entities/   — Entity extraction from text
    POST /api/v3/extract_relations/  — Relation triple extraction from text
"""


# =====================================================================
#  Entity extraction
# =====================================================================

class TestExtractEntities:
    """Entity extraction from a text snippet."""

    def test_valid_text(self, test_client):
        resp = test_client.post(
            "/api/v3/extract_entities/",
            params={"text": "Albert Einstein was born in Ulm."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["extracted_entities"], list)
        assert len(body["extracted_entities"]) > 0

    def test_empty_text_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v3/extract_entities/",
            params={"text": "   "},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_response_echoes_input(self, test_client):
        text = "Python is a programming language."
        resp = test_client.post(
            "/api/v3/extract_entities/",
            params={"text": text},
        )
        assert resp.json()["input_text"] == text


# =====================================================================
#  Relation extraction
# =====================================================================

class TestExtractRelations:
    """Relation (subject-predicate-object triple) extraction."""

    def test_valid_text(self, test_client):
        resp = test_client.post(
            "/api/v3/extract_relations/",
            params={"text": "Marie Curie discovered Radium."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["extracted_relations"], list)

    def test_empty_text_returns_error(self, test_client):
        resp = test_client.post(
            "/api/v3/extract_relations/",
            params={"text": ""},
        )
        assert resp.status_code == 200
        assert "error" in resp.json()

    def test_details_field_present(self, test_client):
        resp = test_client.post(
            "/api/v3/extract_relations/",
            params={"text": "The Earth revolves around the Sun."},
        )
        body = resp.json()
        assert "details" in body
        assert "Extracted" in body["details"]
