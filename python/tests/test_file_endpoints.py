"""Tests for file processing and management endpoints"""

import pytest


def test_file_list_returns_filenames(client):
    """Test retrieving list of indexed files."""
    response = client.get("/file/list")
    assert response.status_code == 200

    response_data = response.json()
    filenames = response_data["filenames"]
    assert isinstance(filenames, list)

def test_file_delete_removes_all_chunks():
    """Test deleting all chunks associated with a file."""
    raise NotImplementedError

def test_file_upload_creates_multiple_docs():
    """Test uploading a file and creating document chunks."""
    raise NotImplementedError
