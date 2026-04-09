import os
import re
from dotenv import load_dotenv
from langsmith import traceable
from langchain_community.document_loaders import PyPDFLoader
import whisper
import asyncio
from functools import partial

from app.models.schemas import ChunkRecord
from app.pipeline.chuncking import Chuncking
from app.db.chroma_client import ChromaClient

load_dotenv(override=False)


class IngestionPipeline:
    def __init__(self):
        self.db_collection = ChromaClient.get_collection()
        self.chunker = Chuncking()
        self.model = None  # Lazy-loaded only for audio ingestion

    def _get_audio_model(self):
        if self.model is None:
            self.model = whisper.load_model("medium")
        return self.model

    def _safe_doc_id(self, file_name: str) -> str:
        stem = os.path.splitext(file_name)[0]
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_")
        return safe or "document"

    @traceable(name="Ingest Document", run_type="tool", save_result=True, use_cache=True)
    async def process_document(self, file_path: str):
        filename = os.path.basename(file_path)
        doc_id = self._safe_doc_id(filename)

        print(f"Starting ingestion for: {filename}")

        # Check for duplicate
        existing = self.db_collection.get(where={"document_id": doc_id}, limit=1)
        if existing["ids"]:
            print(f"Document {doc_id} already exists, skipping.")
            return {"status": "skipped", "doc_id": doc_id, "reason": "Already ingested"}

        # Load PDF in thread pool (blocking I/O)
        loop = asyncio.get_event_loop()
        raw_pages = await loop.run_in_executor(None, self._load_pdf_safe, file_path)

        if not raw_pages:
            raise ValueError("PDF contains no readable pages.")

        # Chunk in thread pool (CPU-bound)
        text_chunks = await loop.run_in_executor(None, self.chunker.chunk_file, raw_pages)
        if not text_chunks:
            raise ValueError("No chunks could be generated from PDF content.")

        ids_to_save, docs_to_save, metas_to_save = [], [], []

        for index, chunk in enumerate(text_chunks):
            chunk_text = (chunk.page_content or "").strip()
            if not chunk_text:
                continue

            chunk_id = f"{doc_id}_CH{index}"
            record = ChunkRecord(
                chunk_id=chunk_id,
                document_id=doc_id,
                text=chunk_text,
                source=filename,
                start_time=None,
                end_time=None,
                page_number=chunk.metadata.get("page", 0),
                metadata={}
            )
            ids_to_save.append(record.chunk_id)
            docs_to_save.append(record.text)
            metas_to_save.append(record.to_persistence_payload())

        if not ids_to_save:
            raise ValueError("PDF parsed, but produced only empty chunks.")

        # Upsert instead of add — safe for re-ingestion
        await loop.run_in_executor(
            None,
            partial(self.db_collection.upsert,
                    ids=ids_to_save,
                    documents=docs_to_save,
                    metadatas=metas_to_save)
        )

        print(f"Saved {len(ids_to_save)} chunks for {doc_id}.")
        return {"status": "success", "doc_id": doc_id, "chunks": len(ids_to_save)}

    @traceable(name="Process Audio file", run_type="tool", save_result=True)
    def transcribe_audio(self, file_path: str) -> dict:
        model = self._get_audio_model()
        results = model.transcribe(file_path)
        file_name = os.path.basename(file_path)
        doc_id = self._safe_doc_id(file_name)

        ids_to_save = []
        docs_to_save = []
        metas_to_save = []

        for seg in results.get("segments", []):
            text = (seg.get("text") or "").strip()
            if not text:
                continue

            record = ChunkRecord(
                chunk_id=f"{doc_id}_CH{seg['id']}",
                document_id=doc_id,
                text=text,
                source=file_name,
                page_number=0,
                start_time=float(seg["start"]),
                end_time=float(seg["end"])
            )

            ids_to_save.append(record.chunk_id)
            docs_to_save.append(record.text)
            metas_to_save.append(record.to_persistence_payload())

        if ids_to_save:
            self.db_collection.add(
                ids=ids_to_save,
                documents=docs_to_save,
                metadatas=metas_to_save
            )
            print(f"Saved {len(ids_to_save)} chunks to persistent storage.")

        return {
            "status": "success",
            "doc_id": doc_id,
            "chunks": len(ids_to_save)
        }