"""Shared test fixtures for VectorForge test suite"""

import pytest

from fastapi.testclient import TestClient

from vectorforge.api import app, engine


@pytest.fixture
def client():
    """Create fresh TestClient for each test"""
    return TestClient(app)

@pytest.fixture(autouse=True)
def reset_engine():
    """Clear the engine state before each test"""
    engine.documents.clear()
    engine.embeddings.clear()
    engine.index_to_doc_id.clear()
    engine.doc_id_to_index.clear()
    engine.deleted_docs.clear()
    yield

@pytest.fixture
def added_doc(client):
    """Fixture that adds a document for search tests."""
    response = client.post("/doc/add", json={
        "content": "Machine learning is fascinating",
        "metadata": {"topic": "AI"}
    })
    
    return response.json()
