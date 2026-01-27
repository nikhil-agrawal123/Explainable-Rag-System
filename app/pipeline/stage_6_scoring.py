# Stage 6: Document Scoring and Transparency
# - Calculate contribution % per document
# - Identify unused documents
import re
from typing import List, Dict
from collections import defaultdict
from app.models.schemas import ChunkRecord
from langsmith import traceable

class TrustScorer:
    def __init__(self):
        # Regex to find patterns like [DataForge_CH5] or [Doc_Name_CH12]
        self.citation_pattern = r"\[(.*?)\]"

    @traceable(name="Stage6_TrustScoring", run_type="tool")
    def calculate_scores(self, retrieved_chunks: List[ChunkRecord], final_answer: str) -> Dict:
        """
        Analyzes the answer to quantify trust and document contribution.
        """
        print("📊 Calculating Trust Scores...")

        # 1. Identify Valid IDs
        valid_chunk_map = {c.chunk_id: c.document_id for c in retrieved_chunks}
        
        # 2. Extract Citations from Answer
        cited_ids = re.findall(self.citation_pattern, final_answer)
        
        # Filter: Only keep IDs that are actually in our retrieved set
        valid_citations = [cid for cid in cited_ids if cid in valid_chunk_map]
        hallucinated_citations = [cid for cid in cited_ids if cid not in valid_chunk_map]

        # 3. Calculate Document Contribution
        doc_counts = defaultdict(int)
        for cid in valid_citations:
            doc_id = valid_chunk_map[cid]
            doc_counts[doc_id] += 1

        # 4. Generate Report
        total_retrieved = len(retrieved_chunks)
        total_used = len(valid_citations)
        
        # Avoid division by zero
        utilization_rate = (total_used / total_retrieved * 100) if total_retrieved > 0 else 0

        # Build the Document Breakdown
        doc_breakdown = []
        unique_docs = set(valid_chunk_map.values())
        
        for doc_id in unique_docs:
            used_count = doc_counts[doc_id]
            contribution = (used_count / total_used * 100) if total_used > 0 else 0
            
            doc_breakdown.append({
                "document_id": doc_id,
                "citations": used_count,
                "contribution_percent": round(contribution, 1)
            })

        return {
            "trust_metrics": {
                "utilization_rate": f"{utilization_rate:.1f}%",
                "total_citations": total_used,
                "hallucinations": len(hallucinated_citations)
            },
            "document_breakdown": doc_breakdown,
            "verification_status": "HIGH" if not hallucinated_citations and total_used > 0 else "LOW"
        }