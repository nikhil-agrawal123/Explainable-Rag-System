"""API router composition.

This module preserves the historical `router` import path while delegating
endpoints to feature-focused route files.
"""

from fastapi import APIRouter

from app.api.routes.ingestion import router as ingestion_router
from app.api.routes.llm_tools import router as llm_router
from app.api.routes.rag import router as rag_router

router = APIRouter()
router.include_router(ingestion_router)
router.include_router(llm_router)
router.include_router(rag_router)
