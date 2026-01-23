# API Routes definition
# POST /ingest
# POST /query (The main RAG endpoint)
import os
import shutil
from typing import List
from fastapi import APIRouter, UploadFile, File
from app.core.config import settings
from app.pipeline.stage_1_ingestion import IngestionPipeline
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv(override=False)

router = APIRouter()

@traceable(name="Ingest Documents Endpoint", run_type="api", save_result=False, use_cache=False)
@router.post("/ingest/", summary="Upload and Process Multiple PDFs")
async def ingest_documents(files: List[UploadFile] = File(...)):    
    # Initialize the worker (loads models once)
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