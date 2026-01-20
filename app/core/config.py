# Configuration settings
# Environment variables (OPENAI_KEY, CHROMA_PATH, etc.)
import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    # Core App Settings
    PROJECT_NAME: str = "DataForge"
    ENVIRONMENT: str = "local"

    # LangChain / LangSmith (Orchestration + Tracing)
    LANGCHAIN_TRACING_V2: bool = Field(default=False)
    LANGCHAIN_API_KEY: Optional[str] = Field(default=None)
    LANGCHAIN_PROJECT: str = Field(default="dataforge")

    # LangSmith
    LANGSMITH_TRACING: bool = Field(default=False)
    LANGSMITH_ENDPOINT: Optional[str] = Field(default=None)
    LANGSMITH_API_KEY: Optional[str] = Field(default=None)
    LANGSMITH_PROJECT: str = Field(default="dataforge")

    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(default=None)

    # HuggingFace
    HUGGINGFACE_API_KEY: Optional[str] = Field(default=None)

    # Qdrant (optional) — keep your existing QDRANT_* names
    QDRANT_API_KEY: Optional[str] = Field(default=None)
    QDRANT_CLUSTER_ENDPOINT: Optional[str] = Field(default=None)

    # Persistence Paths
    CHROMA_PERSIST_DIR: str = Field(default="./data/chroma_storage")
    UPLOAD_DIR: str = Field(default="./data/uploads")
    COLLECTION_NAME: str = Field(default="dataforge_knowledge")


settings = Settings()

# Ensure directories exist immediately upon import
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)