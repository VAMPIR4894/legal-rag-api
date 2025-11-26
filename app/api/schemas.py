from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Data Schemas ---

class SourceMetadata(BaseModel):
    """Schema for the metadata of a retrieved document."""
    # Category/type removed: metadata should include source-specific fields only
    source: Optional[str] = Field(None, description="Section or rule number or source identifier.")
    chapter: Optional[str] = Field(None, description="Chapter number (if applicable).")
    year: Optional[int] = Field(None, description="Year (if available).")
    court: Optional[str] = Field(None, description="Court or issuing body (if available).")

    class Config:
        extra = 'ignore'


class RetrievedSource(BaseModel):
    """Schema for a single document used as context."""
    source_id: str = Field(description="Unique ID used for citation (e.g., S1).")
    text: str = Field(description="The full text chunk used from the source.")
    metadata: SourceMetadata
    rerank_score: float = Field(description="The final relevance score after reranking.")
    source_db: str = Field(description="The collection the document originated from.")


# --- API Request/Response Schemas ---

class QueryRequest(BaseModel):
    """Schema for the API query input."""
    query: str = Field(..., min_length=5, description="The legal question to be answered.")
    
    # --- THIS IS THE NEWLY ADDED FIELD ---
    pdf_context: Optional[str] = Field(None, description="Optional PDF text to use as context, bypassing retrieval.")
    # --- END OF NEW FIELD ---


class QueryResponse(BaseModel):
    """
    Schema for the final, structured API response. 
    (Structure defined fully in Phase 6, but established here.)
    """
    query: str
    answer_with_explanation: str = Field(description="Comprehensive answer that includes both the direct answer and explanation with source citations.")
    sources: List[RetrievedSource] = Field(description="The list of documents used as context.")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    status: str = Field(default="success")
    
    class Config:
        # Allows conversion from dict which may contain keys not defined here
        extra = 'ignore'