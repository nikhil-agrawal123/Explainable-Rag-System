# FastAPI entry point
# Initialize App
# Include API routers
from fastapi import FastAPI
from app.core.config import settings
from app.api.endpoints import router as api_router
from app.db.chroma_client import ChromaClient

app = FastAPI(
    title=settings.PROJECT_NAME, 
    version="1.0",
    description="DataForge RAG: Multi-File Ingestion Engine"
)

# Include the router we made above
app.include_router(api_router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Check database connection on startup"""
    try:
        ChromaClient.get_instance()
        print(f"DataForge Server Online")
        print(f"   Storage: {settings.CHROMA_PERSIST_DIR}")
    except Exception as e:
        print(f"Critical Error: Database not accessible. {e}")

@app.get("/")
def root():
    return {"message": "DataForge Server is Running. Go to /docs for the UI."}