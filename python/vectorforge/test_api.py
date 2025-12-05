from fastapi.testclient import TestClient
from api import app


# =============================================================================
# Setup Configuration 
# =============================================================================
client = TestClient(app)


# =============================================================================
# Endpoint Tests 
# =============================================================================
def test_list_files_endpoint():
    response = client.get("/file/list")
    assert response.status_code == 200

    response_data = response.json()
    filenames = response_data["filenames"]
    assert isinstance(filenames, list)

def test_upload_file_endpoint():
    pass

def test_delete_file_endpoint():
    pass

def test_get_doc_endpoint():
    doc_data = {
        "content": "This is a test document content",
        "metadata": {
            "source_file": "test.txt",
            "author": "test_user"
        }
    }
    # Test add endpoint with dummy doc
    add_response = client.post("/doc/add", json=doc_data)
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
    assert get_response_data["content"] == "This is a test document content"
    assert get_response_data["metadata"]["source_file"] == "test.txt"
    assert get_response_data["metadata"]["author"] == "test_user"

def test_add_doc_endpoint():
    pass

def test_delete_doc_endpoint():
    pass

def test_search_endpoint():
    pass

def test_index_stats_endpoint():
    pass

def test_build_index_endpoint():
    pass

def test_save_index_endpoint():
    pass

def test_load_index_endpoint():
    pass

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    
    response_data = response.json()
    assert response_data["status"] == "healthy"

def test_metrics_endpoint():
    pass
