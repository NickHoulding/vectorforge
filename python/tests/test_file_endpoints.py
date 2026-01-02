"""Tests for file processing and management endpoints"""

import io

import pytest

from vectorforge.config import Config


# =============================================================================
# File Test Fixtures
# =============================================================================

@pytest.fixture
def uploaded_file(client):
    """Upload a single test file and return its metadata.
    
    Creates a test file upload and returns the API response
    containing the document IDs and chunk information.
    
    Returns:
        resp: the response object from the file upload POST request.
    """
    filename = "test_document.txt"
    file_content = b"This is a test document with some content for testing file uploads."
    
    files = {"file": (filename, io.BytesIO(file_content), "text/plain")}
    resp = client.post("/file/upload", files=files)
    
    return resp


# =============================================================================
# File Endpoint Tests
# =============================================================================

def test_file_list_returns_200(client):
    """Test that GET /file/list returns 200 status."""
    response = client.get("/file/list")
    assert response.status_code == 200


def test_file_list_returns_filenames_list(client):
    """Test that file list response contains filenames list."""
    response = client.get("/file/list")
    data = response.json()
    assert "filenames" in data
    assert isinstance(data["filenames"], list)


def test_file_list_empty_when_no_files_uploaded(client):
    """Test that file list is empty when no files have been uploaded."""
    resp = client.get("/file/list")
    assert resp.status_code == 200
    
    filenames = resp.json()["filenames"]
    assert len(filenames) == 0


def test_file_upload_pdf_returns_201(client):
    """Test that uploading a PDF file returns 201 status."""
    filename = "test_document.pdf"
    file_content = b"This is a test document with some content for testing file uploads."
    files = {"file": (filename, io.BytesIO(file_content), "text/pdf")}

    resp = client.post("/file/upload", params=files)
    assert resp.status_code == 201


def test_file_upload_txt_returns_201(uploaded_file):
    """Test that uploading a TXT file returns 201 status."""
    assert uploaded_file.status_code == 201


def test_file_upload_creates_multiple_chunks(client):
    """Test that uploading a file creates multiple document chunks."""
    filename = "test_document.md"
    file_content = b"a" * (Config.DEFAULT_CHUNK_SIZE * 2)
    files = {"file": (filename, io.BytesIO(file_content), "text/plain")}

    resp = client.post("/file/upload", params=files)
    assert resp.status_code == 201
    
    data = resp.json()
    assert len(data["doc_ids"]) > 0


def test_file_upload_returns_chunk_count(uploaded_file):
    """Test that file upload response includes the number of chunks created."""
    data = uploaded_file.json()
    
    assert "chunks_created" in data
    assert isinstance(data["chunks_created"], int)


def test_file_upload_returns_doc_ids(uploaded_file):
    """Test that file upload response includes all document ID strings."""
    data = uploaded_file.json()

    assert "doc_ids" in data
    assert isinstance(data["doc_ids"], list)
    assert all(isinstance(doc_id, str) for doc_id in data["doc_ids"])


def test_file_upload_chunks_have_metadata(client, uploaded_file):
    """Test that uploaded file chunks contain source_file and chunk_index metadata."""
    doc_ids = uploaded_file.json()["doc_ids"]
    for doc_id in doc_ids:
        resp = client.get(f"/doc/{doc_id}")
        assert resp.status_code == 200
        
        doc_data = resp.json()
        assert "source_file" in doc_data
        assert isinstance(doc_data["metadata"]["source_file"], str)
        assert "chunk_index" in doc_data
        assert isinstance(doc_data["metadata"]["chunk_index"], str)


def test_file_upload_unsupported_format_returns_400(client):
    """Test that uploading an unsupported file format returns 400."""
    filename = "test_document.md"
    file_content = b"This is a test document with some content for testing file uploads."
    files = {"file": (filename, io.BytesIO(file_content), "text/unsupported-filetype")}

    resp = client.post("/file/upload", files=files)
    assert resp.status_code == 400


def test_file_upload_no_filename_returns_400(client):
    """Test that uploading without a filename returns 400."""
    file_content = b"This is a test document with some content for testing file uploads."
    files = {"file": (io.BytesIO(file_content), "text/plain")}

    resp = client.post("/file/upload", files=files)
    assert resp.status_code == 400


def test_file_upload_empty_file(client):
    """Test uploading an empty file."""
    filename = "test_document.txt"
    files = {"file": (filename, io.BytesIO(b""), "text/plain")}

    resp = client.post("/file/upload", files=files)
    assert resp.status_code == 400


def test_file_delete_returns_200(client, uploaded_file):
    """Test that file DELETE returns 200 for existing file."""
    resp = client.delete(f"/file/delete/{uploaded_file['filename']}")
    assert resp.status_code == 200


def test_file_delete_removes_all_chunks(client, uploaded_file):
    """Test that deleting a file removes all associated chunks."""
    resp = client.delete(f"/file/delete/{uploaded_file['filename']}")
    assert resp.status_code == 200

    doc_ids = resp.json()["doc_ids"]
    for doc_id in doc_ids:
        resp = client.get(f"/doc/{doc_id}")
        assert resp.status_code == 404


def test_file_delete_returns_404_for_nonexistent_file(client):
    """Test that deleting a non-existent file returns 404."""
    resp = client.delete(f"/file/delete/nonexistant-file.txt")
    assert resp.status_code == 404


def test_file_delete_response_contains_filename(client, uploaded_file):
    """Test that delete response includes the filename."""
    resp = client.delete(f"/file/delete/{uploaded_file['filename']}")
    assert resp.status_code == 200

    data = resp.json()
    assert "filename" in data
    assert isinstance(data["filename"], str)


def test_file_delete_response_contains_chunks_deleted_count(client, uploaded_file):
    """Test that delete response includes count of chunks deleted."""
    resp = client.delete(f"/file/delete/{uploaded_file['filename']}")
    assert resp.status_code == 200

    data = resp.json()
    assert "chunks_deleted" in data
    assert isinstance(data["chunks_deleted"], int)


def test_file_delete_response_contains_doc_ids(client, uploaded_file):
    """Test that delete response includes list of deleted document IDs."""
    resp = client.delete(f"/file/delete/{uploaded_file['filename']}")
    assert resp.status_code == 200

    data = resp.json()
    assert "doc_ids" in data
    assert isinstance(data["doc_ids"], list)
    assert all(isinstance(doc_id, str) for doc_id in data["doc_ids"])


def test_file_list_includes_uploaded_filenames(client, uploaded_file):
    """Test that file list includes filenames of uploaded files."""
    filename = uploaded_file.json()["filename"]

    resp = client.get("/file/list")
    assert resp.status_code == 200

    assert filename in resp.json()["filenames"]


def test_file_list_excludes_deleted_files(client, uploaded_file):
    """Test that file list doesn't include deleted files."""
    filename = uploaded_file.json()["filename"]

    resp = client.delete(f"/file/delete/{filename}")
    assert resp.status_code == 200

    resp = client.get("/file/list")
    assert resp.status_code == 200

    assert filename not in resp.json()["filenames"]


def test_file_upload_with_duplicate_filename(client):
    """Test uploading a file with the same filename as an existing file."""
    raise NotImplementedError


def test_file_delete_does_not_affect_other_files(client):
    """Test that deleting one file doesn't affect other uploaded files."""
    raise NotImplementedError


def test_file_upload_response_format(client):
    """Test that file upload response contains all required fields."""
    raise NotImplementedError


def test_file_upload_large_pdf(client):
    """Test uploading a large PDF file."""
    raise NotImplementedError


def test_file_upload_pdf_with_special_characters_in_filename(client):
    """Test uploading a file with special characters in filename."""
    raise NotImplementedError


def test_file_delete_with_special_characters_in_filename(client):
    """Test deleting a file with special characters in filename."""
    raise NotImplementedError


def test_file_upload_pdf_extracts_text(client):
    """Test that PDF upload correctly extracts text content."""
    raise NotImplementedError


def test_file_upload_txt_decodes_utf8(client):
    """Test that TXT upload correctly decodes UTF-8 content."""
    raise NotImplementedError


def test_file_upload_increments_files_uploaded_metric(client):
    """Test that file upload increments files_uploaded metric."""
    raise NotImplementedError


def test_file_upload_increments_chunks_created_metric(client):
    """Test that file upload increments chunks_created metric."""
    raise NotImplementedError


def test_file_delete_updates_deleted_docs_metric(client):
    """Test that file deletion updates docs_deleted metric."""
    raise NotImplementedError


def test_file_upload_response_contains_status(client):
    """Test that upload response contains 'status' field."""
    raise NotImplementedError


def test_file_upload_response_contains_filename(client):
    """Test that upload response contains filename."""
    raise NotImplementedError


def test_file_delete_response_contains_status(client):
    """Test that delete response contains 'status' field."""
    raise NotImplementedError


def test_file_upload_with_case_variant_extension(client):
    """Test uploading files with .PDF or .TXT (uppercase) extension."""
    raise NotImplementedError
