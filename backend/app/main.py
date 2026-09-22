import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure backend root is on sys.path for direct module execution
current_dir = Path(__file__).resolve().parent
backend_root = current_dir.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.core.config import settings
    from app.services.retrieval import RetrievalService
    from app.services.generation import GenerationService
    from app.api.routes.query import router as query_router
    from app.utils.logging_config import setup_logging
except ImportError:
    from backend.app.core.config import settings
    from backend.app.services.retrieval import RetrievalService
    from backend.app.services.generation import GenerationService
    from backend.app.api.routes.query import router as query_router
    from backend.app.utils.logging_config import setup_logging

logger = setup_logging(settings.LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Loads the vector store and LLM connection once at startup, not on every request.
    """
    logger.info("Starting up Textbook RAG Backend...")
    resolved_store_dir = settings.get_resolved_vector_store_dir()
    logger.info(f"Target vector store directory: {resolved_store_dir}")

    # Load vector store and retrieval service
    app.state.retrieval_service = RetrievalService(
        vector_store_dir=resolved_store_dir,
        collection_name=settings.COLLECTION_NAME,
        embedding_model_name=settings.EMBEDDING_MODEL_NAME,
        k=settings.RETRIEVAL_K,
    )

    # Load Ollama generation service
    app.state.generation_service = GenerationService(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.OLLAMA_MODEL,
    )

    yield

    logger.info("Shutting down Textbook RAG Backend...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="FastAPI service for grounded, citation-backed QA using ChromaDB and Ollama LLM.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS middleware allowing frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else [settings.CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount primary routes (both at root level and versioned prefix)
app.include_router(query_router)
app.include_router(query_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["General"])
async def root():
    """Root status endpoint providing API documentation link."""
    return {
        "service": settings.PROJECT_NAME,
        "status": "running",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "query": "/query",
        },
    }
