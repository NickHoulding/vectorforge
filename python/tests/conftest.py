"""Shared test fixtures for VectorForge test suite"""

import pytest

from fastapi.testclient import TestClient

from vectorforge.api import app, engine


@pytest.fixture
def client():
    """Create fresh TestClient for each test.
    
    Provides a FastAPI TestClient instance for making HTTP requests
    to the VectorForge API endpoints in tests.
    """
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_engine():
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
def added_doc(client):
    """Add a sample document and return its metadata.
    
    Creates a test document about machine learning and returns the API
    response containing the document ID and status.
    """
    response = client.post("/doc/add", json={
        "content": "Machine learning is fascinating",
        "metadata": {"topic": "AI"}
    })
    
    return response.json()
