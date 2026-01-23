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

router = APIRouter()

@router.post("/ingest/", summary="Upload and Process Multiple PDFs")
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

@router.post("/query_decomposition/", summary="Decompose User Query")
async def query_decomposition(query: str):
    # from app.pipeline.stage_2_decomposition import QueryDecompositionPipeline

    decomposition_pipeline = QueryDecompositionPipeline()
    sub_queries = decomposition_pipeline.decompose_query(query)

    return {"original_query": query, "sub_queries": sub_queries}