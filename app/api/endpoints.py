# API Routes definition
# POST /ingest
# POST /query (The main RAG endpoint)
import os
import shutil
from typing import List
from fastapi import APIRouter, UploadFile, File
from app.core.config import settings
from app.pipeline.stage_1_ingestion import IngestionPipeline
from app.pipeline.stage_2_decomposition import QueryDecompositionPipeline
from app.pipeline.stage_3_retrieval import MultiQueryRetrievalPipeline
from app.pipeline.stage_4_local_graph import KnowledgeGraphBuilder
from app.pipeline.stage_5_generation import GenerationEngine
from app.pipeline.stage_6_scoring import TrustScorer
from app.utils.llm import extract_relations, extract_entities
from app.utils.audioIngestion import AudioIngestion
from app.utils.visualizer import GraphVisualizer

router = APIRouter()

@router.post("/ingest_pdf/", summary="Upload and Process Multiple PDFs")
async def ingest_documents(files: List[UploadFile] = File(...)):    
    pipeline = IngestionPipeline()
    
    results = []
    
    print(f"Received {len(files)} files for ingestion.")

    for file in files:
        # 1. Validation
        if not file.filename.endswith(".pdf"):
            results.append({"file": file.filename, "status": "skipped", "reason": "Not a PDF"})
            continue

        # 2. Save to Temp Disk
        temp_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # 3. Trigger the Pipeline
            process_stats = await pipeline.process_document(temp_path)
            
            results.append({
                "file": file.filename,
                "status": "success",
                "details": process_stats
            })

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({
                "file": file.filename, 
                "status": "failed", 
                "error": str(e)
            })
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return {"job_summary": results}

@router.post("/ingest_audio/", summary="Upload and Process Audio File")
async def ingest_audio(file: UploadFile = File(...)):
    # 1. Validation
    if not file.filename.endswith((".mp3", ".wav", ".m4a", ".flac")):
        return {"file": file.filename, "status": "skipped", "reason": "Unsupported audio format"}

    # 2. Save to Temp Disk
    temp_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 3. Transcribe Audio
        audio_ingestion = AudioIngestion("large")
        transcript = audio_ingestion.transcribe_audio(temp_path)
        return {"file": file.filename, "status": "success", "transcript": transcript}
    
    except Exception as e:
        return {"file": file.filename, "status": "failed", "error": str(e)}

@router.post("/extract_entities/", summary="Extract Entities from Text Snippet")
async def extract_entities_endpoint(text: str):
    try:
        if not text or len(text.strip()) == 0:
            return {"error": "Input text cannot be empty."}
        else:
            entities = extract_entities(text)
        return {
            "input_text": text,
            "extracted_entities": entities,
            "details": f"Extracted {len(entities)} entities.",
            "status": "success"
            }

    except Exception as e:
        return {"error": f"Error extracting entities: {e}"}

@router.post("/extract_relations/", summary="Extract Relations from Text Snippet")
async def extract_relations_endpoint(text: str):
    try:
        if not text or len(text.strip()) == 0:
            return {"error": "Input text cannot be empty."}
        else:
            relations = extract_relations(text)
        return {
            "input_text": text,
            "extracted_relations": relations,
            "details": f"Extracted {len(relations)} relations.",
            "status": "success"
            }

    except Exception as e:
        return {"error": f"Error extracting relations: {e}"}

@router.post("/query_decomposition/", summary="Decompose User Query")
async def query_decomposition(query: str):

    try:
        if not query or len(query.strip()) == 0:
            return {"error": "Query cannot be empty."}
        else:
            decomposition_pipeline = QueryDecompositionPipeline()
            sub_queries = decomposition_pipeline.decompose_query(query)
        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "details": f"Decomposed into {len(sub_queries)} sub-queries.",
            "status": "success"
            }

    except Exception as e:
        return {"error": f"Invalid query input: {e}"}

@router.post("/query/", summary="Process User Query through RAG Pipeline")
async def query_rag(query: str):
    try:
        if not query or len(query.strip()) == 0:
            return {"error": "Query cannot be empty."}
        else:
            retrieval_pipeline = MultiQueryRetrievalPipeline()
            decomposition_pipeline = QueryDecompositionPipeline()
            sub_queries = decomposition_pipeline.decompose_query(query)
            chunks = retrieval_pipeline.retrieve_documents(sub_queries, k_per_query=5)
        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "retrieved_chunks": [chunk.dict() for chunk in chunks],
            "details": f"Retrieved {len(chunks)} chunks from {len(sub_queries)} sub-queries.",
            "status": "success"
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
    viz = GraphVisualizer(graph_builder.graph)
    viz.prune_graph(min_edge_weight=1)

    path_2d = viz.generate_2d_html("query_graph_2d.html")
    path_3d = viz.generate_3d_html("query_graph_3d.html")

    return {
        "relational_content": graph_builder.get_relational_context(),
        "stats": {
            "nodes": graph_builder.graph.number_of_nodes(),
            "edges": graph_builder.graph.number_of_edges()
        },
        "2d_graph_path": os.path.abspath(path_2d),
        "3d_graph_path": os.path.abspath(path_3d),
        "status": "success"
    }

@router.post("/full_pipeline/", summary="Run Full RAG Pipeline and Generate Answer")
async def full_pipeline(query: str):
    try:
        if not query or len(query.strip()) == 0:
            return {"error": "Query cannot be empty."}
        
        # Initialize Pipelines
        decomposition_pipeline = QueryDecompositionPipeline()
        retrieval_pipeline = MultiQueryRetrievalPipeline()
        graph_builder = KnowledgeGraphBuilder()
        generation_engine = GenerationEngine()
        
        # 1. Decompose Query
        sub_queries = decomposition_pipeline.decompose_query(query)
        
        # 2. Retrieve Documents
        chunks = retrieval_pipeline.retrieve_documents(sub_queries, k_per_query=5)
        if not chunks:
            return {"error": "No relevant information found for the query."}
        
        # 3. Build Knowledge Graph
        kg = graph_builder.build_graph(chunks)
        graph_builder.prune_graph(min_edge_weight=1)
        graph_text = graph_builder.get_relational_context()

        viz = GraphVisualizer(kg)
        path2d = viz.generate_2d_html("full_pipeline_graph_2d.html")
        path3d = viz.generate_3d_html("full_pipeline_graph_3d.html")
        
        # 4. Generate Answer
        final_answer = generation_engine.generate_answer(query, chunks, graph_text)
        
        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "retrieved_chunks": [chunk.dict() for chunk in chunks],
            "knowledge_graph_stats": {
                "nodes": kg.number_of_nodes(),
                "edges": kg.number_of_edges()
            },
            "final_answer": final_answer,
            "2d_graph_path": os.path.abspath(path2d),
            "3d_graph_path": os.path.abspath(path3d),
            "status": "success"
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
        
        # Retrieve relevant chunks first
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
            "status": "success"
        }
        
    except Exception as e:
        return {"error": f"Error calculating trust scores: {e}"}