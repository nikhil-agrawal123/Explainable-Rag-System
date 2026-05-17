import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import router as api_router
from app.db.chroma_client import ChromaClient

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the application."""
    # ── Startup ──
    try:
        ChromaClient.get_instance()
        logger.info("DataForge Server Online")
        logger.info("   Storage: %s", settings.CHROMA_PERSIST_DIR)
    except Exception as e:
        logger.critical("Database not accessible. %s", e)
    yield
    # ── Shutdown (add cleanup here if needed) ──


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="3.0",
    description="DataForge RAG: Multi-File Ingestion Engine",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/health", summary="Health check endpoint")
def health_check():
    """Simple endpoint to check if the server is running."""
    return {"status": "ok", "message": "DataForge Server is healthy."}

app.include_router(api_router, prefix="/api/v3")

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass