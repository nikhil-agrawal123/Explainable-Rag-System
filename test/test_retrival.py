import sys
import io

# Fix Windows console encoding for Unicode
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from app.pipeline.stage_2_decomposition import QueryDecompositionPipeline
from app.pipeline.stage_3_retrieval import MultiQueryRetrievalPipeline

def main():
    # 1. Setup
    decomposer = QueryDecompositionPipeline() 
    retriever = MultiQueryRetrievalPipeline() 
    
    user_query = "What did Jakob Bernoulli work on?"
    
    # 2. Stage 2: Decompose
    print("\n--- STAGE 2: DECOMPOSITION ---")
    sub_queries = decomposer.decompose_query(user_query)
    print(f"Generated: {sub_queries}")
    
    # 3. Stage 3: Retrieve
    print("\n--- STAGE 3: RETRIEVAL ---")
    chunks = retriever.retrieve_documents(sub_queries, k_per_query=5)
    
    # 4. Verify Output
    print(f"\n--- FINAL CONTEXT ({len(chunks)} chunks) ---")
    for chunk in chunks:
        print(f"\n[ID: {chunk.chunk_id}]")
        print(f"Entities: {chunk.metadata.entities}")
        print(f"Text Snippet: {chunk.text}")

if __name__ == "__main__":
    main()