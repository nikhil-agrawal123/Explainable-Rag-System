# Pydantic Schemas
# Request Models (QueryRequest)
# Response Models (Stage 7: User Visible Output - Answer, Evidence, Graph, etc.)
from pydantic import BaseModel, Field
from typing import List, Optional
import json

#TODO the metadata is not able to store the start and end time for audio chunks. Fix this.

# --- 1. BASE STRUCTURES (The Building Blocks) ---
class Relation(BaseModel):
    subject: str
    predicate: str
    object: str

    def to_list(self) -> List[str]:
        return [self.subject, self.predicate, self.object]

# --- 2. EXTRACTION MODELS (Internal Logic) ---
class ExtractedMetadata(BaseModel):
    entities: List[str] = Field(default_factory=list, description="Key entities found in the chunk")
    relations: List[Relation] = Field(default_factory=list, description="Structured knowledge triples")
    domain: List[str] = Field(default_factory=lambda: ["General"], description="The knowledge domain (e.g., 'Probability Theory')")
# --- 3. INGESTION MODELS (API Input/Output) ---
class ProcessingStats(BaseModel):
    file_name: str
    total_chunks: int
    chunks_with_metadata: int
    status: str

# --- 4. DATABASE MODELS (What actually gets saved) ---
class ChunkRecord(BaseModel):
    chunk_id: str = Field(..., description="Unique ID: DOC_NAME_CH#")
    document_id: str
    text: str
    source: str
    page_number: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None 
    
    metadata: ExtractedMetadata

    class Config:
        from_attributes = True

    def to_persistence_payload(self) -> dict:
        payload =  {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page_number,
            "domain": json.dumps(self.metadata.domain),
            "entities": json.dumps(self.metadata.entities),
            "relations": json.dumps([r.to_list() for r in self.metadata.relations]) 
        }

        if self.start_time is not None:
            payload["start_time"] = self.start_time
        if self.end_time is not None:
            payload["end_time"] = self.end_time
            
        return payload