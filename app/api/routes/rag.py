import os

from fastapi import APIRouter, Depends

from app.api.auth import AuthenticatedUser, get_current_user
from app.core.config import settings
from app.pipeline.stage_2_decomposition import QueryDecompositionPipeline
from app.pipeline.stage_3_retrieval import MultiQueryRetrievalPipeline
from app.pipeline.stage_4_local_graph import KnowledgeGraphBuilder
from app.pipeline.stage_5_generation import GenerationEngine
from app.pipeline.stage_6_scoring import TrustScorer
from app.utils.visualizer import GraphVisualizer

router = APIRouter(tags=["RAG Pipeline"])


def _viz_dir(user_id: str) -> str:
    return os.path.join(settings.VIZ_DIR, user_id)


@router.post("/query/", summary="Process User Query through RAG Pipeline")
async def query_rag(
    query: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        if not query or len(query.strip()) == 0:
            return {"error": "Query cannot be empty."}

        retrieval_pipeline = MultiQueryRetrievalPipeline()
        decomposition_pipeline = QueryDecompositionPipeline()
        sub_queries = decomposition_pipeline.decompose_query(query)
        chunks = retrieval_pipeline.retrieve_documents(sub_queries, k_per_query=5, user_id=current_user.user_id)
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
async def visualize_graph(
    query: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    retrieval_pipeline = MultiQueryRetrievalPipeline()
    graph_builder = KnowledgeGraphBuilder()
    decomposition_pipeline = QueryDecompositionPipeline()
    sub_queries = decomposition_pipeline.decompose_query(query)
    chunks = retrieval_pipeline.retrieve_documents(sub_queries, k_per_query=5, user_id=current_user.user_id)

    graph_builder.build_graph(chunks)
    graph_builder.prune_graph(min_edge_weight=1)
    viz = GraphVisualizer(graph_builder.graph)

    vd = _viz_dir(current_user.user_id)
    path_2d = viz.generate_2d_html(os.path.join(vd, "query_graph_2d.html"))
    path_3d = viz.generate_3d_html(os.path.join(vd, "query_graph_3d.html"))

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
async def full_pipeline(
    query: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        if not query or len(query.strip()) == 0:
            return {"error": "Query cannot be empty."}

        decomposition_pipeline = QueryDecompositionPipeline()
        retrieval_pipeline = MultiQueryRetrievalPipeline()
        graph_builder = KnowledgeGraphBuilder()
        generation_engine = GenerationEngine()

        sub_queries = decomposition_pipeline.decompose_query(query)

        chunks = retrieval_pipeline.retrieve_documents(sub_queries, k_per_query=5, user_id=current_user.user_id)
        if not chunks:
            return {"error": "No relevant information found for the query."}

        kg = graph_builder.build_graph(chunks)
        graph_builder.prune_graph(min_edge_weight=1)
        graph_text = graph_builder.get_relational_context()

        vd = _viz_dir(current_user.user_id)
        viz = GraphVisualizer(kg)
        path2d = viz.generate_2d_html(os.path.join(vd, "full_pipeline_graph_2d.html"))
        path3d = viz.generate_3d_html(os.path.join(vd, "full_pipeline_graph_3d.html"))

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
async def score_trust(
    query: str,
    final_answer: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    try:
        if not query or len(query.strip()) == 0:
            return {"error": "Query cannot be empty."}
        if not final_answer or len(final_answer.strip()) == 0:
            return {"error": "Final answer cannot be empty."}

        decomposition_pipeline = QueryDecompositionPipeline()
        retrieval_pipeline = MultiQueryRetrievalPipeline()
        sub_queries = decomposition_pipeline.decompose_query(query)
        chunks = retrieval_pipeline.retrieve_documents(sub_queries, k_per_query=5, user_id=current_user.user_id)

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
