import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_app():
    """Application factory function."""
    app = FastAPI(
        title="Legal RAG API",
        description="Retrieval-Augmented Generation API for Legal Questions",
        version="1.0.0"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Routes
    from app.api.routes import router
    app.include_router(router, prefix="/api/v1", tags=["RAG"])

    print("FastAPI app initialized. Use /api/v1/health to check status.")
    return app

app = create_app()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)