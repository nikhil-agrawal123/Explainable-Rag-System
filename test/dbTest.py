from app.db.chroma_client import ChromaDBClient
from app.core.config import settings

collection = ChromaDBClient.get_collection()