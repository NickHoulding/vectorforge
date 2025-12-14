"""Tests for system monitoring endpoints"""

import pytest


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    
    response_data = response.json()
    assert response_data["status"] == "healthy"

def test_metrics(client):
    raise NotImplementedError
