"""
Unit tests for core logic modules — no API, no external services.

Tests the pure functions and classes that can be verified without
any mocking at all:
  - TrustScorer.calculate_scores  (regex + math)
  - KnowledgeGraphBuilder.build_graph  (networkx)
  - _extract_json_payload  (string parsing)
"""
from app.pipeline.stage_6_scoring import TrustScorer
from app.pipeline.stage_4_local_graph import KnowledgeGraphBuilder
from app.utils.llm import _extract_json_payload
from app.models.schemas import ChunkRecord, ExtractedMetadata, Relation


# =====================================================================
#  Helper: build sample ChunkRecords
# =====================================================================

def _make_chunks():
    """Create a small set of chunks with known structure."""
    return [
        ChunkRecord(
            chunk_id="Doc1_CH0",
            document_id="Doc1",
            text="Einstein developed relativity.",
            source="physics.pdf",
            page_number=1,
            metadata=ExtractedMetadata(
                entities=["Einstein", "relativity"],
                relations=[Relation(subject="Einstein", predicate="developed", object="relativity")],
                domain=["Physics"],
            ),
        ),
        ChunkRecord(
            chunk_id="Doc1_CH1",
            document_id="Doc1",
            text="Newton discovered gravity.",
            source="physics.pdf",
            page_number=2,
            metadata=ExtractedMetadata(
                entities=["Newton", "gravity"],
                relations=[Relation(subject="Newton", predicate="discovered", object="gravity")],
                domain=["Physics"],
            ),
        ),
        ChunkRecord(
            chunk_id="Doc2_CH0",
            document_id="Doc2",
            text="Python is a programming language.",
            source="cs.pdf",
            page_number=1,
            metadata=ExtractedMetadata(
                entities=["Python"],
                relations=[],
                domain=["Computer Science"],
            ),
        ),
    ]


# =====================================================================
#  TrustScorer — citation extraction & scoring
# =====================================================================

class TestTrustScorer:
    """Unit tests for the real TrustScorer logic."""

    def setup_method(self):
        self.scorer = TrustScorer()
        self.chunks = _make_chunks()

    def test_valid_citations_counted(self):
        """Citations matching retrieved chunk IDs should be counted."""
        answer = "Einstein developed relativity [Doc1_CH0]. Newton discovered gravity [Doc1_CH1]."
        result = self.scorer.calculate_scores(self.chunks, answer)
        assert result["trust_metrics"]["total_citations"] == 2

    def test_hallucinated_citations_detected(self):
        """Citations NOT in retrieved set should be flagged as hallucinations."""
        answer = "Some claim [FAKE_ID]. Real claim [Doc1_CH0]."
        result = self.scorer.calculate_scores(self.chunks, answer)
        assert result["trust_metrics"]["hallucinations"] == 1
        assert result["trust_metrics"]["total_citations"] == 1

    def test_no_citations_gives_low_trust(self):
        """An answer with zero citations should get LOW verification."""
        answer = "Some answer with no citations at all."
        result = self.scorer.calculate_scores(self.chunks, answer)
        assert result["verification_status"] == "LOW"
        assert result["trust_metrics"]["total_citations"] == 0

    def test_all_valid_gives_high_trust(self):
        """All-valid citations with no hallucinations → HIGH."""
        answer = "Claim A [Doc1_CH0]. Claim B [Doc1_CH1]."
        result = self.scorer.calculate_scores(self.chunks, answer)
        assert result["verification_status"] == "HIGH"
        assert result["trust_metrics"]["hallucinations"] == 0

    def test_hallucination_gives_low_trust(self):
        """Any hallucinated citation → LOW verification."""
        answer = "Claim [Doc1_CH0]. Fake [NOT_REAL_CH99]."
        result = self.scorer.calculate_scores(self.chunks, answer)
        assert result["verification_status"] == "LOW"

    def test_document_breakdown_contribution(self):
        """Document contribution percentages should sum correctly."""
        answer = "A [Doc1_CH0]. B [Doc1_CH1]. C [Doc2_CH0]."
        result = self.scorer.calculate_scores(self.chunks, answer)
        breakdown = result["document_breakdown"]
        total_pct = sum(d["contribution_percent"] for d in breakdown)
        assert abs(total_pct - 100.0) < 0.5  # ~100% accounting for rounding

    def test_utilization_rate_format(self):
        """Utilization rate should be a string ending in %."""
        answer = "A [Doc1_CH0]."
        result = self.scorer.calculate_scores(self.chunks, answer)
        assert result["trust_metrics"]["utilization_rate"].endswith("%")

    def test_empty_answer(self):
        """Empty answer → zero citations, LOW trust."""
        result = self.scorer.calculate_scores(self.chunks, "")
        assert result["trust_metrics"]["total_citations"] == 0
        assert result["verification_status"] == "LOW"

    def test_empty_chunks(self):
        """No retrieved chunks → still produces a valid report."""
        result = self.scorer.calculate_scores([], "Some answer [Doc1_CH0].")
        assert result["trust_metrics"]["total_citations"] == 0


# =====================================================================
#  KnowledgeGraphBuilder — graph construction
# =====================================================================

class TestKnowledgeGraphBuilder:
    """Unit tests for the real graph construction logic."""

    def setup_method(self):
        self.builder = KnowledgeGraphBuilder()
        self.chunks = _make_chunks()

    def test_graph_has_correct_node_count(self):
        """All unique entities should appear as nodes."""
        self.builder.build_graph(self.chunks)
        # Entities: Einstein, relativity, Newton, gravity, Python = 5
        assert self.builder.graph.number_of_nodes() == 5

    def test_graph_has_correct_edge_count(self):
        """Each relation triple should create one edge."""
        self.builder.build_graph(self.chunks)
        # Relations: Einstein→relativity, Newton→gravity = 2
        assert self.builder.graph.number_of_edges() == 2

    def test_edges_have_provenance(self):
        """Each edge should carry chunk_id and document_id metadata."""
        self.builder.build_graph(self.chunks)
        for u, v, data in self.builder.graph.edges(data=True):
            assert "chunk_id" in data
            assert "document_id" in data
            assert "source" in data

    def test_relational_context_format(self):
        """get_relational_context() should produce readable text."""
        self.builder.build_graph(self.chunks)
        context = self.builder.get_relational_context()
        assert isinstance(context, str)
        # Should contain arrow notation
        assert "-->" in context or "→" in context or "--[" in context

    def test_graph_stats(self):
        """get_graph_stats() should return correct counts."""
        self.builder.build_graph(self.chunks)
        stats = self.builder.get_graph_stats()
        assert stats["node_count"] == 5
        assert stats["edge_count"] == 2
        assert isinstance(stats["entities"], list)

    def test_clear_on_rebuild(self):
        """Building twice should clear the first graph."""
        self.builder.build_graph(self.chunks)
        assert self.builder.graph.number_of_nodes() == 5
        # Build again with subset
        self.builder.build_graph(self.chunks[:1])
        assert self.builder.graph.number_of_nodes() == 2  # Einstein, relativity

    def test_empty_chunks(self):
        """Empty input should produce an empty graph."""
        self.builder.build_graph([])
        assert self.builder.graph.number_of_nodes() == 0
        assert self.builder.graph.number_of_edges() == 0

    def test_prune_removes_isolated_nodes(self):
        """Pruning with high weight threshold should remove low-weight edges."""
        self.builder.build_graph(self.chunks)
        # All edges have default weight=1, so pruning at min_edge_weight=2
        # should remove all edges and then isolated nodes
        self.builder.prune_graph(min_edge_weight=2)
        assert self.builder.graph.number_of_edges() == 0


# =====================================================================
#  _extract_json_payload — JSON extraction from LLM output
# =====================================================================

class TestExtractJsonPayload:
    """Unit tests for the JSON extraction utility."""

    def test_plain_json(self):
        """Plain JSON string should pass through."""
        result = _extract_json_payload('["a", "b"]')
        assert result == '["a", "b"]'

    def test_json_in_code_block(self):
        """JSON wrapped in ```json fences should be extracted."""
        raw = '```json\n["a", "b"]\n```'
        result = _extract_json_payload(raw)
        assert result == '["a", "b"]'

    def test_json_in_plain_code_block(self):
        """JSON wrapped in ``` fences (no language) should be extracted."""
        raw = '```\n{"key": "value"}\n```'
        result = _extract_json_payload(raw)
        assert result == '{"key": "value"}'

    def test_none_input(self):
        result = _extract_json_payload(None)
        assert result == ""

    def test_empty_string(self):
        result = _extract_json_payload("")
        assert result == ""

    def test_whitespace_only(self):
        result = _extract_json_payload("   ")
        assert result == ""
