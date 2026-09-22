import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

# Ensure backend root is on sys.path
backend_root = Path(__file__).resolve().parent.parent
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.main import app


@pytest.fixture
def client():
    """Create a TestClient with mocked retrieval and generation services."""
    # Mock retrieval service
    mock_retrieval = MagicMock()
    mock_retrieval.is_ready = True
    mock_retrieval.retrieve.return_value = (
        "[Passage 1 | Source: Introduction_to_Python_Programming-WEB.pdf, p.10]\nA list is a mutable sequence.",
        ["Introduction_to_Python_Programming-WEB.pdf, p.10"],
    )

    # Mock generation service
    mock_generation = MagicMock()
    mock_generation.is_connected.return_value = True
    mock_generation.generate_answer.return_value = (
        "A list in Python is a mutable sequence of values [Source: Introduction_to_Python_Programming-WEB.pdf, p.10]."
    )

    with TestClient(app) as test_client:
        # Override services on app.state
        test_client.app.state.retrieval_service = mock_retrieval
        test_client.app.state.generation_service = mock_generation
        yield test_client


def test_health_check(client):
    """Test GET /health returns 200 and expected status schema."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "vector_store_loaded" in data
    assert "ollama_connected" in data
    assert data["status"] == "healthy"


def test_query_happy_path(client):
    """Test POST /query with valid question returns 200 and cited answer."""
    payload = {"question": "What is a Python list?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert len(data["sources"]) > 0
    assert "Introduction_to_Python_Programming-WEB.pdf" in data["sources"][0]


def test_query_invalid_input_empty_payload(client):
    """Test POST /query with missing question returns 422 Unprocessable Entity."""
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_query_invalid_input_empty_string(client):
    """Test POST /query with empty string question returns 422 Unprocessable Entity."""
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 422


def test_query_api_v1_route(client):
    """Test POST /api/v1/query route works identically."""
    payload = {"question": "Explain lists vs tuples"}
    response = client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
