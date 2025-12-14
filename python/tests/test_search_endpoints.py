"""Tests for search endpoints"""

import pytest


def test_search_basic(client, added_doc):
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

def test_search_small_top_k(client):
    """Handle top_k value less than the default (10)"""
    raise NotImplementedError

def test_search_large_top_k(client):
    """Handle top_k value greater than the default (10)"""
    raise NotImplementedError

def test_search_negative_top_k(client):
    """Make sure negative top_k values are disallowed"""
    raise NotImplementedError

def test_search_with_filters(client):
    """Make sure filtered searching works properly"""
    raise NotImplementedError
