import os 
from typing import List
from dotenv import load_dotenv
from langsmith import traceable
from langchain_community.document_loaders import PyPDFLoader

from app.models.schemas import ChunkRecord
from app.pipeline.metadata import MetadataExtractor
from app.pipeline.chuncking import Chuncking
from app.db.chroma_client import ChromaClient

load_dotenv(override=False)

class IngestionPipeline:
    def __init__(self):
        self.db_collection = ChromaClient.get_collection()
        self.chunker = Chuncking()
        self.metadata_extractor = MetadataExtractor()

    @traceable(name="Ingest Document", run_type="tool", save_result=True, use_cache=True)
    async def 