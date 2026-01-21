# Stage 1: Data Storage and Chunking
# - Chunking logic
# - Metadata extraction (Topics, Entities)
# - Prepare chunks for Vector Store
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langsmith import traceable
from typing import List
from dotenv import load_dotenv

load_dotenv(override=False)

class Chuncking():
    @traceable(name="Chuncking Init", run_type="tool", save_result=False, use_cache=False)
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    @traceable(name="Chunking Stage" , save_result=True , use_cache=True , run_type="tool")
    def chunk_file(self, pages:List[Document]) -> List[Document]:
        
        chunks = self.splitter.split_documents(pages)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        return chunks
