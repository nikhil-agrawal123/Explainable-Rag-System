# Pydantic Schemas
# Request Models (QueryRequest)
# Response Models (Stage 7: User Visible Output - Answer, Evidence, Graph, etc.)
from pydantic import BaseModel, Field
from typing import List

# --- 1. BASE STRUCTURES (The Building Blocks) ---
class Relation(BaseModel):
    """
    Represents a knowledge triple: (Subject, Predicate, Object).
    Reference: Source 43-45
    """
    subject: str
    predicate: str
    object: str

    def to_list(self) -> List[str]:
        """Helper to convert to the list format [S, P, O] often used in graphs."""
        return [self.subject, self.predicate, self.object]

# --- 2. EXTRACTION MODELS (Internal Logic) ---
class ExtractedMetadata(BaseModel):
    """
    The output from your Stage 1 'Entity Extraction' (Spacy/LLM).
    This is the clean, structured data BEFORE it hits the database.
    Reference: Source 34, 42-46
    """
    entities: List[str] = Field(default_factory=list, description="Key entities found in the chunk")
    relations: List[Relation] = Field(default_factory=list, description="Structured knowledge triples")
    domain: str = Field(default="General", description="The knowledge domain (e.g., 'Probability Theory')")

# --- 3. INGESTION MODELS (API Input/Output) ---
class ProcessingStats(BaseModel):
    """
    Return metrics to the user after ingestion.
    """
    file_name: str
    total_chunks: int
    chunks_with_metadata: int
    status: str

# --- 4. DATABASE MODELS (What actually gets saved) ---
class ChunkRecord(BaseModel):
    """
    Represents a fully processed chunk ready for the vector store.
    Reference: Source 38-46
    """
    chunk_id: str = Field(..., description="Unique ID: DOC_NAME_CH#")
    document_id: str
    text: str
    source: str
    page_number: int
    
    metadata: ExtractedMetadata

    class Config:
        from_attributes = True

    def to_persistence_payload(self) -> dict:
        """
        Converts the rich structure into the flat format ChromaDB requires.
        Lists are serialized to JSON strings here.
        Reference: Source 35 (Metadata remains unchanged/mappable)
        """
        import json
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page_number,
            "domain": self.metadata.domain,
            "entities": json.dumps(self.metadata.entities),
            "relations": json.dumps([r.to_list() for r in self.metadata.relations]) 
        }