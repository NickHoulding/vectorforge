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


def test_doc_get_returns_200_with_valid_id(client, added_doc):
    """Test that GET /doc/{id} returns 200 for existing document."""
    response = client.get(f"/doc/{added_doc['id']}")
    assert response.status_code == 200

def test_doc_get_returns_correct_id(client, added_doc):
    """Test that retrieved document has correct ID."""
    response = client.get(f"/doc/{added_doc['id']}")
    data = response.json()
    assert data["id"] == added_doc["id"]

def test_doc_get_returns_matching_content(client, sample_doc):
    """Test that retrieved document content matches original."""
    add_response = client.post("/doc/add", json=sample_doc)
    doc_id = add_response.json()["id"]
    
    get_response = client.get(f"/doc/{doc_id}")
    data = get_response.json()
    assert data["content"] == sample_doc["content"]

def test_doc_get_preserves_metadata(client, sample_doc):
    """Test that retrieved document preserves metadata."""
    add_response = client.post("/doc/add", json=sample_doc)
    doc_id = add_response.json()["id"]
    
    get_response = client.get(f"/doc/{doc_id}")
    metadata = get_response.json()["metadata"]
    assert metadata["source_file"] == "test.txt"
    assert metadata["author"] == "test_user"

def test_doc_get_returns_404_when_not_found(client):
    """Test 404 response when retrieving a non-existent document."""
    response = client.get("/doc/nonexistent-id")
    assert response.status_code == 404

def test_doc_add_returns_201(client, sample_doc):
    """Test that POST /doc/add returns 201 status."""
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201

def test_doc_add_returns_document_id(client, sample_doc):
    """Test that POST /doc/add returns a document ID."""
    response = client.post("/doc/add", json=sample_doc)
    data = response.json()
    assert "id" in data
    assert isinstance(data["id"], str)
    assert len(data["id"]) > 0

def test_doc_add_creates_document_with_id(client, sample_doc):
    """Test that POST /doc/add returns indexed status."""
    response = client.post("/doc/add", json=sample_doc)
    data = response.json()
    assert data["status"] == "indexed"

def test_doc_delete_returns_200(client, added_doc):
    """Test that DELETE /doc/{id} returns 200 status."""
    response = client.delete(f"/doc/{added_doc['id']}")
    assert response.status_code == 200

def test_doc_delete_returns_deleted_status(client, added_doc):
    """Test that DELETE /doc/{id} returns deleted status."""
    response = client.delete(f"/doc/{added_doc['id']}")
    data = response.json()
    assert data["status"] == "deleted"

def test_doc_delete_removes_from_index(client, added_doc):
    """Test that deleted document is no longer retrievable."""
    doc_id = added_doc["id"]
    client.delete(f"/doc/{doc_id}")
    
    get_response = client.get(f"/doc/{doc_id}")
    assert get_response.status_code == 404

def test_doc_delete_returns_404_when_not_found(client):
    """Test 404 response when deleting a non-existent document."""
    response = client.delete("/doc/nonexistent-id")
    assert response.status_code == 404
