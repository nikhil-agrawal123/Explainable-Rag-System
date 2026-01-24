from app.models.schemas import ChunkRecord, ExtractedMetadata, Relation
import networkx as nx
from app.pipeline.stage_4_local_graph import KnowledgeGraphBuilder
from app.utils.visualizer import GraphVisualizer

import os

def main():
    # 1. Create Dummy Data (Simulating Retrieval output)
    chunk1 = ChunkRecord(
        chunk_id="DocA_CH1",
        document_id="DocA",
        text="Bernoulli logic...",
        source="test.pdf",
        page_number=1,
        metadata=ExtractedMetadata(
            entities=["Jakob Bernoulli", "Ars Conjectandi"],
            relations=[
                Relation(subject="Jakob Bernoulli", predicate="wrote", object="Ars Conjectandi")
            ]
        )
    )

    chunk2 = ChunkRecord(
        chunk_id="DocB_CH5",
        document_id="DocB",
        text="More math...",
        source="history.pdf",
        page_number=10,
        metadata=ExtractedMetadata(
            entities=["Ars Conjectandi", "1713"],
            relations=[
                Relation(subject="Ars Conjectandi", predicate="published_in", object="1713")
            ]
        )
    )

    # 2. Build Graph
    kg = KnowledgeGraphBuilder()
    graph = kg.build_graph([chunk1, chunk2])

    # 3. Inspect
    print("\n---  Graph Structure ---")
    print(f"Nodes: {graph.nodes()}")
    
    print("\n--- LLM Context View (Triples) ---")
    print(kg.get_relational_context())

    # 5. Visualize
    print("\n--- Visualization ---")
    viz = GraphVisualizer(graph)
    
    # Generate 2D (The useful one)
    path_2d = viz.generate_2d_html("data_forge_2d.html")
    print(f"2D Graph saved to: {os.path.abspath(path_2d)}")
    
    # Generate 3D (The cool one)
    path_3d = viz.generate_3d_html("data_forge_3d.html")
    print(f"3D Graph saved to: {os.path.abspath(path_3d)}")

    # 4. Verify Multi-Hop Potential
    # Does a path exist from Bernoulli to 1713?
    try:
        path = nx.shortest_path(graph, "Jakob Bernoulli", "1713")
        print(f"\nPath found: {path}")
    except:
        print("\nNo path found.")

if __name__ == "__main__":
    main()