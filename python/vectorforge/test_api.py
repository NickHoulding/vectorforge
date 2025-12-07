import pytest
from fastapi.testclient import TestClient
from api import app, engine


# =============================================================================
# Fixtures 
# =============================================================================
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
def sample_doc():
    """Reusable sample document data"""
    return {
        "content": "This is a test document content",
        "metadata": {
            "source_file": "test.txt",
            "author": "test_user"
        }
    }

@pytest.fixture
def added_doc(client, sample_doc):
    """Create and return a document that's already added to the engine"""
    response = client.post("/doc/add", json=sample_doc)
    return {
        "id": response.json()["id"],
        "data": sample_doc
    }


# =============================================================================
# Endpoint Tests 
# =============================================================================
def test_list_files_endpoint(client):
    response = client.get("/file/list")
    assert response.status_code == 200

    response_data = response.json()
    filenames = response_data["filenames"]
    assert isinstance(filenames, list)

def test_upload_file_endpoint():
    raise NotImplementedError

def test_delete_file_endpoint():
    raise NotImplementedError

def test_get_doc_endpoint(client, sample_doc):
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

def test_get_doc_not_found(client):
    """Test getting a non-existent document"""
    response = client.get("/doc/nonexistent-id")
    assert response.status_code == 404

def test_add_doc_endpoint(client, sample_doc):
    response = client.post("/doc/add", json=sample_doc)
    assert response.status_code == 201

    response_data = response.json()
    assert "id" in response_data
    assert response_data["status"] == "indexed"

def test_delete_doc_endpoint(client, added_doc):
    """Test deletion using the added_doc fixture"""
    doc_id = added_doc["id"]

    response = client.delete(f"/doc/{doc_id}")
    assert response.status_code == 200

    response_data = response.json()
    assert response_data["id"] == doc_id
    assert response_data["status"] == "deleted"

    get_response = client.get(f"/doc/{doc_id}")
    assert get_response.status_code == 404

def test_delete_doc_not_found(client):
    """Test deleting a nonexistant doc"""
    response = client.delete("/doc/nonexistent-id")
    assert response.status_code == 404

def test_search_endpoint(client, added_doc):
    """Test searching for the added sample doc"""
    search_query = {
        "query": "test document"
    }

    response = client.post("/search", json=search_query)
    assert response.status_code == 200

    response_data = response.json()
    assert response_data["query"] == "test document"
    assert "results" in response_data
    assert response_data["count"] >= 0

def test_small_top_k(client):
    """Handle top_k value less than the default (10)"""
    raise NotImplementedError

def test_large_top_k(client):
    """Handle top_k value greater than the default (10)"""
    raise NotImplementedError

def test_negative_top_k(client):
    """Make sure negative top_k values are disallowed"""
    raise NotImplementedError

def test_index_stats_endpoint(client):
    raise NotImplementedError

def test_build_index_endpoint(client):
    raise NotImplementedError

def test_save_index_endpoint(client):
    raise NotImplementedError

def test_load_index_endpoint(client):
    raise NotImplementedError

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    
    response_data = response.json()
    assert response_data["status"] == "healthy"

def test_metrics_endpoint(client):
    raise NotImplementedError
