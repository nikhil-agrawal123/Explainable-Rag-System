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
BATCH_SIZE = 16  # Adjust based on your GPU memory

class OllamaEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function using Ollama for ChromaDB with batch processing."""
    
    def __init__(self, model_name: str = OLLAMA_MODEL, batch_size: int = BATCH_SIZE):
        self.model_name = model_name
        self.batch_size = batch_size
    
    def __call__(self, input: Documents) -> Embeddings:
        all_embeddings = []
        
        for i in range(0, len(input), self.batch_size):
            batch = input[i:i + self.batch_size]
            # Ollama supports batch embedding with a list of inputs
            response = ollama.embed(model=self.model_name, input=batch)
            all_embeddings.extend(response['embeddings'])
        
        return all_embeddings

# Create a singleton instance
embedding_function = OllamaEmbeddingFunction()

@traceable(name="Generate_Embeddings", run_type="tool")
def get_embedding_function(texts: list, batch_size: int = BATCH_SIZE) -> list:
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = ollama.embed(model=OLLAMA_MODEL, input=batch)
        all_embeddings.extend(response['embeddings'])
    
    return all_embeddings

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