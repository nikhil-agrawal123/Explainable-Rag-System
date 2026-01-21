from app.db.chroma_client import ChromaClient
from app.core.config import settings

collection = ChromaClient.get_collection()