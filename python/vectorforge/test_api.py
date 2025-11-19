from fastapi.testclient import TestClient
from api import app


# Setup configuration
client = TestClient(app)


# Endpoint tests
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
