"""
API client for communicating with the FastAPI RAG backend.
Loads API_BASE_URL from environment — never hardcodes the backend URL.
"""

import os
import requests
from dotenv import load_dotenv

# Load .env from this file's directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

TIMEOUT_SECONDS = 120  # LLM generation can take time on first call


def check_health() -> dict:
    """
    Call GET /health on the backend.
    Returns a dict with service status or an error message.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        response.raise_for_status()
        return {"ok": True, **response.json()}
    except requests.exceptions.ConnectionError:
        return {
            "ok": False,
            "status": "offline",
            "vector_store_loaded": False,
            "ollama_connected": False,
            "model": "—",
            "error": f"Cannot connect to backend at {API_BASE_URL}",
        }
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "status": "timeout",
            "vector_store_loaded": False,
            "ollama_connected": False,
            "model": "—",
            "error": "Backend health check timed out.",
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "error",
            "vector_store_loaded": False,
            "ollama_connected": False,
            "model": "—",
            "error": str(exc),
        }


def query_assistant(question: str) -> dict:
    """
    Call POST /query on the backend.
    Returns {"answer": str, "sources": list[str]} or raises RuntimeError on failure.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question.strip()},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to backend at {API_BASE_URL}. "
            "Make sure the backend is running: `uvicorn app.main:app --reload` inside `backend/`."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "The backend took too long to respond. "
            "This may happen when Ollama is loading the model for the first time. "
            "Please try again."
        )
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"Backend error: {detail}")
    except Exception as exc:
        raise RuntimeError(f"Unexpected error: {exc}")
