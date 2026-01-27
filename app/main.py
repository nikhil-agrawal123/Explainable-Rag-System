# FastAPI entry point
# Initialize App
# Include API routers
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.core.config import settings
from app.api.endpoints import router as api_router
from app.db.chroma_client import ChromaClient

app = FastAPI(
    title=settings.PROJECT_NAME, 
    version="3.0",
    description="DataForge RAG: Multi-File Ingestion Engine",
    docs_url=None,  # Disable default docs
)

# Mount static files for custom CSS
app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(api_router, prefix="/api/v3")

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui():
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{settings.PROJECT_NAME} - Docs</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
        <link rel="stylesheet" href="/static/swagger-dark.css">
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script>
            SwaggerUIBundle({{
                url: "/openapi.json",
                dom_id: '#swagger-ui',
                presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
                layout: "BaseLayout",
                syntaxHighlight: {{ theme: "obsidian" }}
            }});
        </script>
    </body>
    </html>
    """)

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