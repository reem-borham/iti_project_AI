import json
from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Textbook RAG Assistant"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = "INFO"

    # Ollama LLM Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:1b"

    # Vector Store & Embedding Configuration
    VECTOR_STORE_DIR: str = "./data/vector_store"
    COLLECTION_NAME: str = "rag_textbooks"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    RETRIEVAL_K: int = 4

    # CORS Configuration
    CORS_ORIGINS: Union[List[str], str] = ["http://localhost:8501", "http://localhost:3000", "http://localhost:8000"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("[") and v.endswith("]"):
                try:
                    return json.loads(v)
                except Exception:
                    pass
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    def get_resolved_vector_store_dir(self) -> Path:
        """Resolve vector store directory, checking current and parent locations."""
        direct_path = Path(self.VECTOR_STORE_DIR)
        if direct_path.is_absolute():
            return direct_path
        
        # Check relative to current working directory
        if direct_path.exists():
            return direct_path.resolve()
        
        # Check backend/data/vector_store
        backend_rel = Path(__file__).resolve().parent.parent.parent / "data" / "vector_store"
        if backend_rel.exists():
            return backend_rel.resolve()

        # Check root data/vector_store
        root_rel = Path(__file__).resolve().parent.parent.parent.parent / "data" / "vector_store"
        if root_rel.exists():
            return root_rel.resolve()

        return direct_path.resolve()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
