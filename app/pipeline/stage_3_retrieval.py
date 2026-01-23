# Stage 3: Multi-Query Retrieval
# - Interact with ChromaDB
# - Hybrid Search (Vector + Keyword)
# - Second-stage grader

#TODO implement hybrid search 

from jsonschema._utils import uniq
from app.db.chroma_client import ChromaClient   
from app.pipeline.stage_2_decomposition import QueryDecompositionPipeline
from langsmith import traceable
from dotenv import load_dotenv
from app.models.schemas import ChunkRecord, ExtractedMetadata, Relation
import json

load_dotenv(override=False)

class MultiQueryRetrievalPipeline:
    def __init__(self):
        self.chroma_client = ChromaClient.get_instance()
        self.collection = ChromaClient.get_collection()
        self.decomposition_pipeline = QueryDecompositionPipeline()

    @traceable(name="Multi-Query Retrieval", run_type="tool", save_result=True, use_cache=True)
    def retrieve_documents(self, sub_queries: list[str], k_per_query: int) -> list[ChunkRecord]:

        results = self.collection.query(
            query_texts=sub_queries,
            n_results=k_per_query,
            include=["metadatas", "documents", "distances"]
        )

        unique_chunks = {}
        for q_index, doc_list in enumerate(results["documents"]):
            for i, doc_text in enumerate(doc_list):
                # Extract raw data from Chroma response
                chunk_id = results["ids"][q_index][i]
                raw_meta = results["metadatas"][q_index][i]
                
                # If we already have this chunk, skip it (or maybe boost its score later)
                if chunk_id in unique_chunks:
                    continue

                # 3. RECONSTRUCT OBJECTS
                try:
                    # Parse domain - it's stored as JSON string in ChromaDB
                    domain_raw = raw_meta.get("domain", '["General"]')
                    if isinstance(domain_raw, str):
                        domain = json.loads(domain_raw)
                    else:
                        domain = domain_raw if isinstance(domain_raw, list) else ["General"]
                    
                    rich_meta = ExtractedMetadata(
                        domain=domain,
                        entities=json.loads(raw_meta.get("entities", "[]")),
                        relations=[
                            Relation(subject=r[0], predicate=r[1], object=r[2]) 
                            for r in json.loads(raw_meta.get("relations", "[]"))
                        ]
                    )
                except Exception as e:
                    print(f"Metadata Parse Error for {chunk_id}: {e}")
                    rich_meta = ExtractedMetadata()

                # Build the Record
                record = ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=raw_meta.get("document_id", "unknown"),
                    text=doc_text,
                    source=raw_meta.get("source", "unknown"),
                    page_number=raw_meta.get("page", 0),
                    metadata=rich_meta
                )

                unique_chunks[chunk_id] = record

        unique_list = list(unique_chunks.values())
        print(f"Retrieved {len(unique_list)} unique chunks from {len(sub_queries)} queries.")
        return unique_list

