from fastapi.testclient import TestClient
from api import app


# =============================================================================
# Setup Configuration 
# =============================================================================
client = TestClient(app)


# =============================================================================
# Endpoint Tests 
# =============================================================================
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

def test_list_files_endpoint():
    response = client.get("/file/list")
    assert response.status_code == 200

    response_data = response.json()
    filenames = response_data["filenames"]
    assert isinstance(filenames, list)
