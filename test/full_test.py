import asyncio
import time

# Import all your hard work
from app.pipeline.stage_2_decomposition import QueryDecompositionPipeline as QueryDecomposer
from app.pipeline.stage_3_retrieval import MultiQueryRetrievalPipeline as RetrievalEngine
from app.pipeline.stage_4_local_graph import KnowledgeGraphBuilder
from app.pipeline.stage_5_generation import GenerationEngine
from app.utils.visualizer import GraphVisualizer

async def main():
    # --- 0. SETUP ---
    print( "INITIALIZING SYSTEM...")
    decomposer = QueryDecomposer()
    retriever = RetrievalEngine()
    graph_builder = KnowledgeGraphBuilder()
    generator = GenerationEngine()
    
    # User Query
    query = "What are the contributions of Jakob Bernoulli?"
    print(f"\n\nUSER QUERY: {query}")
    print("="*60)

    start_time = time.time()

    # --- STAGE 2: DECOMPOSITION ---
    sub_queries = decomposer.decompose_query(query)
    print(f"\nSUB-QUERIES: {sub_queries}")

    # --- STAGE 3: RETRIEVAL (Includes Lazy Extraction) ---
    chunks = retriever.retrieve_documents(sub_queries, k_per_query=5)
    if not chunks:
        print("No relevant information found.")
        return

    # --- STAGE 4: GRAPH CONSTRUCTION ---
    kg = graph_builder.build_graph(chunks)
    graph_builder.prune_graph(min_edge_weight=1) # Clean up noise
    
    # Generate Graph Context text for the LLM
    graph_text = graph_builder.get_relational_context()

    # --- STAGE 5: GENERATION ---
    print("\nWRITING FINAL ANSWER...")
    final_answer = generator.generate_answer(query, chunks, graph_text)
    
    end_time = time.time()

    # --- FINAL OUTPUT ---
    print("\n" + "="*60)
    print("DATAFORGE ANSWER")
    print("="*60)
    print(final_answer)
    print("="*60)
    
    print(f"\nTotal Time: {end_time - start_time:.2f}s")
    
    # Optional: Save Graph Visualization
    viz = GraphVisualizer(kg)
    viz.generate_2d_html("final_graph.html")
    viz.generate_3d_html("final_graph_3d.html")
    print(f"Graph saved to 'final_graph.html'")

if __name__ == "__main__":
    asyncio.run(main())