import os 
from dotenv import load_dotenv
from langsmith import traceable
from langchain_community.document_loaders import PyPDFLoader

from app.models.schemas import ChunkRecord
from app.pipeline.chuncking import Chuncking
from app.db.chroma_client import ChromaClient

load_dotenv(override=False)

class IngestionPipeline:
    def __init__(self):
        self.db_collection = ChromaClient.get_collection()
        self.chunker = Chuncking()

    @traceable(name="Ingest Document", run_type="tool", save_result=True, use_cache=True)
    async def process_document(self, file_path: str):
        filename = os.path.basename(file_path)
        doc_id = filename.replace(".pdf", "").replace(" ", "_")

        print(f"Starting ingestion for: {filename}")

        # --- STEP A: LOAD ---
        loader = PyPDFLoader(file_path)
        raw_pages = loader.load()

        # --- STEP B: CHUNK ---
        text_chunks = self.chunker.chunk_file(raw_pages)
        print(f"Generated {len(text_chunks)} chunks.")

        # --- STEP C: ENRICH (The Loop) ---
        ids_to_save = []
        docs_to_save = []
        metas_to_save = []

        for index, chunk in enumerate(text_chunks):
            # 1. Generate Deterministic ID
            chunk_id = f"{doc_id}_CH{index}"

            # 2. Have the metadata on the retrieval pipeline to optimize gpu usage
            rich_metadata = {}

            # 3. Validate & Structure
            record = ChunkRecord(
                chunk_id=chunk_id,
                document_id=doc_id,
                text=chunk.page_content,
                source=filename,
                page_number=chunk.metadata.get("page", 0),
                metadata=rich_metadata
            )

            # 4. Prepare for Storage
            ids_to_save.append(record.chunk_id)
            docs_to_save.append(record.text)
            metas_to_save.append(record.to_persistence_payload())

        # --- STEP D: SAVE ---
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