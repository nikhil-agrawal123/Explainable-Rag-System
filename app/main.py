# FastAPI entry point
# Initialize App
# Include API routers
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app.core.config import settings
from app.api.endpoints import router as api_router
from app.db.chroma_client import ChromaClient
import sys


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the application."""
    # ── Startup ──
    try:
        ChromaClient.get_instance()
        print(f"DataForge Server Online")
        print(f"   Storage: {settings.CHROMA_PERSIST_DIR}")
    except Exception as e:
        print(f"Critical Error: Database not accessible. {e}")
    yield
    # ── Shutdown (add cleanup here if needed) ──


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="3.0",
    description="DataForge RAG: Multi-File Ingestion Engine",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v3")


@app.get("/")
def root():
    return {"message": "DataForge Server is Running. Go to /docs for the UI."}

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass