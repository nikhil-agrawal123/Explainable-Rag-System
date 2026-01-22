# ChromaDB Persistent Client setup
# - Initialize PersistentClient
# - Collection management
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from app.core.config import settings
from langsmith import traceable
import ollama
from dotenv import load_dotenv

load_dotenv(override=False)

OLLAMA_MODEL = "qwen3-embedding:8b"  

class OllamaEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using Ollama for ChromaDB."""
    
    def __init__(self, model_name: str = OLLAMA_MODEL):
        self.model_name = model_name
    
    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            response = ollama.embed(model=self.model_name, input=text)
            embeddings.append(response['embeddings'][0])
        return embeddings

# Create a singleton instance
embedding_function = OllamaEmbeddingFunction()

@traceable(name="Generate_Embeddings", run_type="tool")
def get_embedding_function(texts: list):
    """Generate embeddings for a list of texts using Ollama Qwen3."""
    embeddings = []
    for text in texts:
        response = ollama.embed(model=OLLAMA_MODEL, input=text)
        embeddings.append(response['embeddings'][0])
    return embeddings

class ChromaClient:
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
        return client.get_or_create_collection(
            name=name,
            embedding_function=embedding_function
        )

# --- Dependency Injection for FastAPI ---
def get_vector_db():
    return ChromaClient.get_collection()