import logging
import os
import shutil
from typing import List

from fastapi import APIRouter, Depends, File, UploadFile

from app.api.auth import AuthenticatedUser, get_current_user
from app.core.config import settings
from app.pipeline.stage_1_ingestion import IngestionPipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ingestion"])


@router.post("/ingest_pdf/", summary="Upload and Process Multiple PDFs")
async def ingest_documents(
    files: List[UploadFile] = File(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    pipeline = IngestionPipeline()
    results = []
    user_id = current_user.user_id
    user_upload_dir = os.path.join(settings.UPLOAD_DIR, user_id)
    os.makedirs(user_upload_dir, exist_ok=True)

    logger.info("Received %d files for ingestion (user=%s).", len(files), user_id)

    for file in files:
        if not file.filename.endswith(".pdf"):
            results.append({"file": file.filename, "status": "skipped", "reason": "Not a PDF"})
            continue

        temp_path = os.path.join(user_upload_dir, file.filename)
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            process_stats = await pipeline.process_document(temp_path, user_id=user_id)

            results.append({
                "file": file.filename,
                "status": "success",
                "details": process_stats,
            })

        except Exception as e:
            logger.error("Error processing %s: %s", file.filename, e)
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
async def ingest_audio(
    file: UploadFile = File(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    if not file.filename.endswith((".mp3", ".wav", ".m4a", ".flac")):
        return {"file": file.filename, "status": "skipped", "reason": "Unsupported audio format"}

    user_id = current_user.user_id
    user_upload_dir = os.path.join(settings.UPLOAD_DIR, user_id)
    os.makedirs(user_upload_dir, exist_ok=True)
    temp_path = os.path.join(user_upload_dir, file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        audio_ingestion = IngestionPipeline()
        transcript = audio_ingestion.transcribe_audio(temp_path, user_id=user_id)
        return {"file": file.filename, "status": "success", "transcript": transcript}

    except Exception as e:
        return {"file": file.filename, "status": "failed", "error": str(e)}
