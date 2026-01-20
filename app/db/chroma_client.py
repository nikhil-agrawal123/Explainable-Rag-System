# ChromaDB Persistent Client setup
# - Initialize PersistentClient
# - Collection management

import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings
from langsmith import traceable

class ChromaDBClient:
    _instance = None

    @classmethod
    @traceable(name="Get_ChromaDB_Instance", run_type="tool")
    def get_instance(cls):
        if cls._instance is None:
            print(f"[Orchestrator] Mounting ChromaDB at: {settings.CHROMA_PERSIST_DIR}")
            cls._instance = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(allow_reset=True)
            )
        return cls._instance

    @classmethod
    @traceable(name="Get_Collection", run_type="tool") 
    def get_collection(cls, name: str = settings.COLLECTION_NAME):
        client = cls.get_instance()
        return client.get_or_create_collection(name)

# --- Dependency Injection for FastAPI ---
def get_vector_db():
    return ChromaDBClient.get_collection()