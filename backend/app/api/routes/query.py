import logging
from fastapi import APIRouter, HTTPException, Request, status

try:
    from app.schemas.query import HealthResponse, QueryRequest, QueryResponse
    from app.core.config import settings
except ImportError:
    from backend.app.schemas.query import HealthResponse, QueryRequest, QueryResponse
    from backend.app.core.config import settings

logger = logging.getLogger("rag_backend.routes")

router = APIRouter(tags=["RAG Assistant"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the operational health of the backend, vector store, and LLM connection.",
)
async def health_check(request: Request) -> HealthResponse:
    """Return health status of vector store and Ollama services."""
    retrieval_service = getattr(request.app.state, "retrieval_service", None)
    generation_service = getattr(request.app.state, "generation_service", None)

    vs_ready = bool(retrieval_service and retrieval_service.is_ready)
    ollama_ready = bool(generation_service and generation_service.is_connected())

    return HealthResponse(
        status="healthy" if vs_ready else "degraded",
        vector_store_loaded=vs_ready,
        ollama_connected=ollama_ready,
        model=settings.OLLAMA_MODEL,
        collection=settings.COLLECTION_NAME,
    )


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query RAG Assistant",
    description="Retrieve relevant textbook passages and generate a grounded, cited answer.",
)
async def query_assistant(payload: QueryRequest, request: Request) -> QueryResponse:
    """Execute end-to-end RAG: retrieve chunks -> build prompt -> generate cited answer."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty.",
        )

    retrieval_service = getattr(request.app.state, "retrieval_service", None)
    generation_service = getattr(request.app.state, "generation_service", None)

    if not retrieval_service or not generation_service:
        logger.error("Core services not initialized in application state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG services are not initialized.",
        )

    logger.info(f"Processing query: {question}")
    
    # 1. Retrieve relevant passages
    context, sources = retrieval_service.retrieve(question)

    # 2. Generate cited answer via Ollama LLM
    answer = generation_service.generate_answer(question=question, context=context)

    return QueryResponse(
        answer=answer,
        sources=sources,
    )
