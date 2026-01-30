# Stage 3: Multi-Query Retrieval
# - Interact with ChromaDB
# - Hybrid Search (Vector + Keyword)
# - Second-stage grader

#TODO implement hybrid search 

from app.db.chroma_client import ChromaClient   
from app.pipeline.stage_2_decomposition import QueryDecompositionPipeline
from app.pipeline.metadata import MetadataExtractor
from langsmith import traceable
from dotenv import load_dotenv
from app.models.schemas import ChunkRecord

load_dotenv(override=False)

class MultiQueryRetrievalPipeline:
    def __init__(self):
        self.collection = ChromaClient.get_collection()
        self.decomposition_pipeline = QueryDecompositionPipeline()
        self.metadata_extractor = MetadataExtractor()

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
                
                if chunk_id in unique_chunks:
                    continue

                rich_meta = self.metadata_extractor.extract_metadata(doc_text)

                # Build the Record
                record = ChunkRecord(
                    chunk_id=chunk_id,
                    document_id=raw_meta.get("document_id", "unknown"),
                    text=doc_text,
                    source=raw_meta.get("source", "unknown"),
                    page_number=raw_meta.get("page", 0),
                    start_time=raw_meta.get("start_time"),  # Audio timestamps
                    end_time=raw_meta.get("end_time"),      # Audio timestamps
                    metadata=rich_meta
                )

                unique_chunks[chunk_id] = record

        unique_list = list(unique_chunks.values())
        print(f"Retrieved {len(unique_list)} unique chunks from {len(sub_queries)} queries.")
        return unique_list

