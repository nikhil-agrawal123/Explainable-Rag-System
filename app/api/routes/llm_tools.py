from fastapi import APIRouter

from app.pipeline.stage_2_decomposition import QueryDecompositionPipeline
from app.utils.llm import extract_entities, extract_relations

router = APIRouter()


@router.post("/extract_entities/", summary="Extract Entities from Text Snippet")
async def extract_entities_endpoint(text: str):
    try:
        if not text or len(text.strip()) == 0:
            return {"error": "Input text cannot be empty."}

        entities = extract_entities(text)
        return {
            "input_text": text,
            "extracted_entities": entities,
            "details": f"Extracted {len(entities)} entities.",
            "status": "success",
        }

    except Exception as e:
        return {"error": f"Error extracting entities: {e}"}


@router.post("/extract_relations/", summary="Extract Relations from Text Snippet")
async def extract_relations_endpoint(text: str):
    try:
        if not text or len(text.strip()) == 0:
            return {"error": "Input text cannot be empty."}

        relations = extract_relations(text)
        return {
            "input_text": text,
            "extracted_relations": relations,
            "details": f"Extracted {len(relations)} relations.",
            "status": "success",
        }

    except Exception as e:
        return {"error": f"Error extracting relations: {e}"}


@router.post("/query_decomposition/", summary="Decompose User Query")
async def query_decomposition(query: str):
    try:
        if not query or len(query.strip()) == 0:
            return {"error": "Query cannot be empty."}

        decomposition_pipeline = QueryDecompositionPipeline()
        sub_queries = decomposition_pipeline.decompose_query(query)
        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "details": f"Decomposed into {len(sub_queries)} sub-queries.",
            "status": "success",
        }

    except Exception as e:
        return {"error": f"Invalid query input: {e}"}
