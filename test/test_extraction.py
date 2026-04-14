"""
Tests for the NLP extraction endpoints.

    POST /api/v3/extract_entities/   — Entity extraction from text
    POST /api/v3/extract_relations/  — Relation triple extraction from text

These tests exercise the REAL JSON parsing logic in app/utils/llm.py.
The FakeLLM returns structured JSON strings, and the real
_extract_json_payload / json.loads code is tested.
"""


# =====================================================================
#  Entity extraction
# =====================================================================

class TestExtractEntities:
    """Entity extraction from a text snippet (real parsing, fake LLM)."""

    def test_valid_text_returns_entities(self, test_client):
        """FakeLLM returns JSON array → real parser should extract entities."""
        resp = test_client.post(
            "/api/v3/extract_entities/",
            params={"text": "Albert Einstein was born in Ulm."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["extracted_entities"], list)
        # FakeLLM returns ["Entity_A", "Entity_B", "Entity_C"]
        # The real extract_entities() parses this JSON — verify it worked
        assert len(body["extracted_entities"]) == 3
        assert "Entity_A" in body["extracted_entities"]

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

    def test_entities_are_strings(self, test_client):
        """Verify that the real parser returns a list of strings, not nested."""
        resp = test_client.post(
            "/api/v3/extract_entities/",
            params={"text": "Test entity extraction pipeline."},
        )
        body = resp.json()
        for entity in body["extracted_entities"]:
            assert isinstance(entity, str)


# =====================================================================
#  Relation extraction
# =====================================================================

class TestExtractRelations:
    """Relation (subject-predicate-object triple) extraction."""

    def test_valid_text_returns_triples(self, test_client):
        """FakeLLM returns JSON triples → real parser extracts them."""
        resp = test_client.post(
            "/api/v3/extract_relations/",
            params={"text": "Marie Curie discovered Radium."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["extracted_relations"], list)
        # FakeLLM returns 2 relation triples — verify real parsing worked
        assert len(body["extracted_relations"]) == 2

    def test_each_relation_is_triple(self, test_client):
        """Each relation should be a list of exactly 3 strings."""
        resp = test_client.post(
            "/api/v3/extract_relations/",
            params={"text": "The Earth revolves around the Sun."},
        )
        body = resp.json()
        for rel in body["extracted_relations"]:
            assert isinstance(rel, list)
            assert len(rel) == 3
            for part in rel:
                assert isinstance(part, str)

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
