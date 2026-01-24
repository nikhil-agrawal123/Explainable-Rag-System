# Stage 4: The local graph creation
# - Entity extraction from retrieved chunks
# - Relation extraction (Triples)
# - Construct local evidence graph
import networkx as nx
from typing import List, Dict, Any
from app.models.schemas import ChunkRecord
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv(override=False)

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    @traceable(name="Stage4_GraphConstruction", run_type="chain")
    def build_graph(self, chunks: List[ChunkRecord]) -> nx.MultiDiGraph:
        """
        Turns flat chunks into a rich network of entities and relations.
        """
        print(f" Building Graph from {len(chunks)} chunks...")
        
        # Clear previous state if reusing the object
        self.graph.clear()

        for chunk in chunks:
            # 1. Add Entities as Nodes
            for entity in chunk.metadata.entities:
                # We can store attributes on nodes if needed (e.g., type, domain)
                self.graph.add_node(entity, type="entity", domain=chunk.metadata.domain)

            # 2. Add Relations as Edges
            for relation in chunk.metadata.relations:
                # Add the edge with RICH provenance metadata
                self.graph.add_edge(
                    relation.subject,
                    relation.object,
                    relation=relation.predicate,
                    chunk_id=chunk.chunk_id,      # <--- The Citation Key
                    document_id=chunk.document_id, # <--- For Stage 6 Scoring
                    source=chunk.source
                )

        print(f" Graph Built: {self.graph.number_of_nodes()} Nodes, {self.graph.number_of_edges()} Edges")
        return self.graph

    def get_relational_context(self) -> str:
        """
        Converts the graph into a text format the LLM can read (Triples).
        Format: Subject --[predicate]--> Object (Source: DocID)
        """
        context_lines = []
        
        for u, v, data in self.graph.edges(data=True):
            line = f"({u}) --[{data['relation']}]--> ({v}) [Ref: {data['chunk_id']}]"
            context_lines.append(line)
            
        return "\n".join(context_lines)


    @traceable(name="Graph Stats", run_type="tool", save_result=True, use_cache=True)
    def get_graph_stats(self) -> Dict[str, Any]:
        """Returns simple stats for the API response"""
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "entities": list(self.graph.nodes())
        }