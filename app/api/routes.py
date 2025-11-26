from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import ValidationError
import time
import os
import sys
import secrets
import traceback

# Adjust path to import RAGPipeline and Schemas
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'app')) 

# Phase 4/5/6 Imports
from rag.pipeline import RAGPipeline
from rag.response_parser import ResponseParser
from app.api.schemas import QueryRequest, QueryResponse

# Initialize the Router
router = APIRouter()
security = HTTPBasic()

# --- Global Initialization (Loads Models) ---
try:
    RAG_PIPELINE = RAGPipeline()
    RESPONSE_PARSER = ResponseParser()
    print("RAG Pipeline initialized successfully!")
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize RAG Pipeline: {e}")
    traceback.print_exc()
    RAG_PIPELINE = None
    RESPONSE_PARSER = None

# Basic Auth Configuration
USER = os.environ.get("BASIC_AUTH_USER", "legal_user")
PASS = os.environ.get("BASIC_AUTH_PASS", "super_secure_password123")

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Verify basic authentication credentials."""
    correct_username = secrets.compare_digest(credentials.username, USER)
    correct_password = secrets.compare_digest(credentials.password, PASS)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@router.get('/health')
def health_check():
    """Returns the system status."""
    status = "healthy" if RAG_PIPELINE is not None else "degraded"
    return {
        "status": status,
        "timestamp": time.time(),
        "model": os.environ.get("OLLAMA_MODEL_NAME", "mistral-json-rag-v1"),
        "message": "Legal RAG API is operational."
    }

@router.post('/query', response_model=QueryResponse)
def rag_query(request: QueryRequest, username: str = Depends(verify_credentials)):
    """
    Runs the full RAG pipeline or context-only generation for a given query.
    """
    if RAG_PIPELINE is None:
        raise HTTPException(status_code=503, detail="RAG pipeline failed to initialize during startup.")

    try:
        query = request.query
        pdf_context = request.pdf_context
        
        start_time = time.time()
        
        # Routing logic based on whether pdf_context is provided
        if pdf_context:
            # PDF text is provided - skip retrieval
            print("--- Running context-only generation (PDF text provided) ---")
            generated_text, final_docs = RAG_PIPELINE.run_generation_with_context(query, pdf_context)
        else:
            # No PDF text - run full RAG pipeline with database retrieval
            print("--- Running full RAG pipeline (no PDF context) ---")
            generated_text, final_docs = RAG_PIPELINE.run_rag_pipeline(query)
            
        # Parse and Format Response
        if RESPONSE_PARSER is None:
            raise HTTPException(status_code=500, detail="Response parser failed to initialize.")
             
        parsed_output = RESPONSE_PARSER.parse_llm_output(generated_text)
        
        # Construct Final Response
        response_data = QueryResponse(
            query=query,
            answer_with_explanation=parsed_output["answer_with_explanation"],
            sources=final_docs 
        )

        duration = time.time() - start_time
        print(f"Query '{query[:20]}...' completed in {duration:.2f}s.")

        return response_data

    except HTTPException:
        # Re-raise HTTP exceptions
        raise 
    except Exception as e:
        # Catch any unexpected RAG runtime errors
        print(f"RAG Pipeline Runtime Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal processing error: {type(e).__name__}")