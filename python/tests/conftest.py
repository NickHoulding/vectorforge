"""Shared test fixtures for VectorForge test suite"""

from typing import Any, Generator

import pytest

from fastapi.testclient import TestClient
from httpx import Response

from vectorforge.api import app, engine


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def anyio_backend() -> str:
    """Configure anyio to use only asyncio backend."""
    return "asyncio"


@pytest.fixture
def client() -> TestClient:
    """Create fresh TestClient for each test.
    
    Provides a FastAPI TestClient instance for making HTTP requests
    to the VectorForge API endpoints in tests.
    """
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_engine() -> Generator[None, Any, None]:
    """Clear the engine state before each test.
    
    Automatically runs before every test to ensure a clean slate.
    Clears all documents, embeddings, and index mappings.
    """
    engine.documents.clear()
    engine.embeddings.clear()
    engine.index_to_doc_id.clear()
    engine.doc_id_to_index.clear()
    engine.deleted_docs.clear()
    yield


@pytest.fixture
def sample_doc() -> dict[str, Any]:
    """Reusable sample document data"""
    return {
        "content": "This is a test document content",
        "metadata": {
            "source_file": "test.txt",
            "chunk_index": 0
        }
    }


@pytest.fixture
def added_doc(client: TestClient) -> Any:
    """Add a sample document and return its metadata.
    
    Creates a test document about machine learning and returns the API
    response containing the document ID and status.
    """
    response: Response = client.post("/doc/add", json={
        "content": "Machine learning is fascinating",
        "metadata": {
            "topic": "AI"
        }
    })
    
    return response.json()


@pytest.fixture
def multiple_added_docs(client: TestClient) -> list[str]:
    """Add multiple documents with varied content for similarity testing.
    
    Creates 20 documents with diverse topics to ensure different similarity
    scores when searching. Useful for testing score ordering, top_k, etc.
    
    Returns:
        list[str]: List of document IDs
    """
    varied_content: list[str] = [
        "Python is a high-level programming language used for web development",
        "Machine learning algorithms can predict patterns in data",
        "The solar system contains eight planets orbiting the sun",
        "Classical music compositions from the baroque period are timeless",
        "Healthy eating habits include consuming fruits and vegetables daily",
        "Climate change is affecting global weather patterns significantly",
        "Ancient civilizations built impressive architectural structures",
        "Quantum physics explores the behavior of subatomic particles",
        "Professional sports require rigorous training and dedication",
        "Modern art movements challenged traditional painting techniques",
        "Database systems store and organize large amounts of information",
        "Shakespeare wrote numerous plays during the Elizabethan era",
        "Volcanic eruptions can dramatically alter landscapes and ecosystems",
        "Economic theories attempt to explain market behavior and trends",
        "Photography techniques have evolved with digital technology",
        "Marine biology studies organisms living in ocean environments",
        "Renaissance architecture featured symmetry and classical elements",
        "Chemical reactions involve the transformation of molecular structures",
        "Jazz music originated in African American communities",
        "Cybersecurity protects computer systems from malicious attacks"
    ]
    
    doc_ids: list[str] = []
    for i, content in enumerate(varied_content):
        response: Response = client.post("/doc/add", json={
            "content": content,
            "metadata": {
                "source_file": f"doc_{i}.txt",
                "chunk_index": 0
            }
        })
        doc_ids.append(response.json()["id"])
    
    return doc_ids
