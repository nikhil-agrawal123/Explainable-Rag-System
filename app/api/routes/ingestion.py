import os
import shutil
from typing import List

from fastapi import APIRouter, File, UploadFile

from app.core.config import settings
from app.pipeline.stage_1_ingestion import IngestionPipeline

router = APIRouter()


@router.post("/ingest_pdf/", summary="Upload and Process Multiple PDFs")
async def ingest_documents(files: List[UploadFile] = File(...)):
    pipeline = IngestionPipeline()
    results = []

    print(f"Received {len(files)} files for ingestion.")

    for file in files:
        if not file.filename.endswith(".pdf"):
            results.append({"file": file.filename, "status": "skipped", "reason": "Not a PDF"})
            continue

        temp_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            process_stats = await pipeline.process_document(temp_path)

            results.append({
                "file": file.filename,
                "status": "success",
                "details": process_stats,
            })

        except Exception as e:
            print(f"Error processing {file.filename}: {e}")
            results.append({
                "file": file.filename,
                "status": "failed",
                "error": str(e),
            })

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    return {"job_summary": results}


@router.post("/ingest_audio/", summary="Upload and Process Audio File")
async def ingest_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp3", ".wav", ".m4a", ".flac")):
        return {"file": file.filename, "status": "skipped", "reason": "Unsupported audio format"}

    temp_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        audio_ingestion = IngestionPipeline()
        transcript = audio_ingestion.transcribe_audio(temp_path)
        return {"file": file.filename, "status": "success", "transcript": transcript}

    except Exception as e:
        return {"file": file.filename, "status": "failed", "error": str(e)}
