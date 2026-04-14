import os

from fastapi import APIRouter

from app.pipeline.stage_2_decomposition import QueryDecompositionPipeline
from app.pipeline.stage_3_retrieval import MultiQueryRetrievalPipeline
from app.pipeline.stage_4_local_graph import KnowledgeGraphBuilder
from app.pipeline.stage_5_generation import GenerationEngine
from app.pipeline.stage_6_scoring import TrustScorer
from app.utils.visualizer import GraphVisualizer

router = APIRouter()


@router.post("/query/", summary="Process User Query through RAG Pipeline")
async def query_rag(query: str):
    try:
        if not query or len(query.strip()) == 0:
            return {"error": "Query cannot be empty."}

        retrieval_pipeline = MultiQueryRetrievalPipeline()
        decomposition_pipeline = QueryDecompositionPipeline()
        sub_queries = decomposition_pipeline.decompose_query(query)
        chunks = retrieval_pipeline.retrieve_documents(sub_queries, k_per_query=5)
        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "retrieved_chunks": [chunk.model_dump() for chunk in chunks],
            "details": f"Retrieved {len(chunks)} chunks from {len(sub_queries)} sub-queries.",
            "status": "success",
        }
    except Exception as e:
        return {"error": f"Error processing query: {e}"}


@router.post("/visualize_graph/", summary="Visualize Knowledge Graph")
async def visualize_graph(query: str):
    retrieval_pipeline = MultiQueryRetrievalPipeline()
    graph_builder = KnowledgeGraphBuilder()
    decomposition_pipeline = QueryDecompositionPipeline()
    sub_queries = decomposition_pipeline.decompose_query(query)
    chunks = retrieval_pipeline.retrieve_documents(sub_queries, k_per_query=5)

    graph_builder.build_graph(chunks)
    graph_builder.prune_graph(min_edge_weight=1)
    viz = GraphVisualizer(graph_builder.graph)

    path_2d = viz.generate_2d_html("query_graph_2d.html")
    path_3d = viz.generate_3d_html("query_graph_3d.html")

    return {
        "relational_content": graph_builder.get_relational_context(),
        "stats": {
            "nodes": graph_builder.graph.number_of_nodes(),
            "edges": graph_builder.graph.number_of_edges(),
        },
        "2d_graph_path": os.path.abspath(path_2d),
        "3d_graph_path": os.path.abspath(path_3d),
        "status": "success",
    }


@router.post("/full_pipeline/", summary="Run Full RAG Pipeline and Generate Answer")
async def full_pipeline(query: str):
    try:
        if not query or len(query.strip()) == 0:
            return {"error": "Query cannot be empty."}

        decomposition_pipeline = QueryDecompositionPipeline()
        retrieval_pipeline = MultiQueryRetrievalPipeline()
        graph_builder = KnowledgeGraphBuilder()
        generation_engine = GenerationEngine()

        sub_queries = decomposition_pipeline.decompose_query(query)

        chunks = retrieval_pipeline.retrieve_documents(sub_queries, k_per_query=5)
        if not chunks:
            return {"error": "No relevant information found for the query."}

        kg = graph_builder.build_graph(chunks)
        graph_builder.prune_graph(min_edge_weight=1)
        graph_text = graph_builder.get_relational_context()

        viz = GraphVisualizer(kg)
        path2d = viz.generate_2d_html("full_pipeline_graph_2d.html")
        path3d = viz.generate_3d_html("full_pipeline_graph_3d.html")

        final_answer = generation_engine.generate_answer(query, chunks, graph_text)

        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "retrieved_chunks": [chunk.model_dump() for chunk in chunks],
            "knowledge_graph_stats": {
                "nodes": kg.number_of_nodes(),
                "edges": kg.number_of_edges(),
            },
            "final_answer": final_answer,
            "2d_graph_path": os.path.abspath(path2d),
            "3d_graph_path": os.path.abspath(path3d),
            "status": "success",
        }

    except Exception as e:
        return {"error": f"Error in full pipeline: {e}"}


@router.post("/score_trust/", summary="Calculate Trust Scores for Generated Answer")
async def score_trust(query: str, final_answer: str):
    try:
        if not query or len(query.strip()) == 0:
            return {"error": "Query cannot be empty."}
        if not final_answer or len(final_answer.strip()) == 0:
            return {"error": "Final answer cannot be empty."}

        decomposition_pipeline = QueryDecompositionPipeline()
        retrieval_pipeline = MultiQueryRetrievalPipeline()
        sub_queries = decomposition_pipeline.decompose_query(query)
        chunks = retrieval_pipeline.retrieve_documents(sub_queries, k_per_query=5)

        if not chunks:
            return {"error": "No relevant documents found to score trust."}

        scorer = TrustScorer()
        trust_report = scorer.calculate_scores(chunks, final_answer)

        return {
            "original_query": query,
            "final_answer": final_answer,
            "trust_report": trust_report,
            "status": "success",
        }

    except Exception as e:
        return {"error": f"Error calculating trust scores: {e}"}
