"""Tests for document management endpoints"""

import pytest


@pytest.fixture
def sample_doc():
    """Reusable sample document data"""
    return {
        "content": "This is a test document content",
        "metadata": {
            "source_file": "test.txt",
            "author": "test_user"
        }
    }


def test_doc_get_returns_matching_content(client, sample_doc):
    """Test retrieving a document by ID.
    
    Adds a document, retrieves it by ID, and verifies that the returned
    content and metadata match the original document.
    """
    # Test add endpoint with dummy doc
    add_response = client.post("/doc/add", json=sample_doc)
    assert add_response.status_code == 201

    # Check response for correct values
    add_response_data = add_response.json()
    assert isinstance(add_response_data["id"], str)
    assert isinstance(add_response_data["status"], str)
    assert add_response_data["status"] == "indexed"

    # Test get endpoint with the same doc
    doc_id = add_response_data["id"]
    get_response = client.get(f"/doc/{doc_id}")
    assert get_response.status_code == 200
    
    # Check response for correct values
    get_response_data = get_response.json()
    assert get_response_data["id"] == doc_id
    assert get_response_data["content"] == sample_doc["content"]
    assert get_response_data["metadata"]["source_file"] == "test.txt"
    assert get_response_data["metadata"]["author"] == "test_user"

def test_doc_get_returns_404_when_not_found(client):
    """Test 404 response when retrieving a non-existent document."""
    response = client.get("/doc/nonexistent-id")
    assert response.status_code == 404

def test_doc_add_creates_document_with_id(client, sample_doc):
    """Test adding a new document via POST /doc/add."""
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201

    response_data = response.json()
    assert "id" in response_data
    assert response_data["status"] == "indexed"

def test_doc_delete_removes_from_index(client, added_doc):
    """Test deleting a document and verifying it's no longer retrievable."""
    doc_id = added_doc["id"]

    response = client.delete(f"/doc/{doc_id}")
    assert response.status_code == 200

    response_data = response.json()
    assert response_data["id"] == doc_id
    assert response_data["status"] == "deleted"

    get_response = client.get(f"/doc/{doc_id}")
    assert get_response.status_code == 404

def test_doc_delete_returns_404_when_not_found(client):
    """Test 404 response when deleting a non-existent document."""
    response = client.delete("/doc/nonexistent-id")
    assert response.status_code == 404
