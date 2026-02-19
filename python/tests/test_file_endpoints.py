"""Tests for file processing and management endpoints"""

import io

import pytest

from vectorforge.config import VFGConfig

# =============================================================================
# File Test Fixtures
# =============================================================================


@pytest.fixture
def sample_file():
    """Return a sample text file for testing uploads."""
    return (
        "test_document.txt",
        b"This is a test document with some content for testing file uploads.",
    )


@pytest.fixture
def upload_file(client):
    """Factory fixture to upload a file and return response data."""

    def _upload(filename, content, mime_type="text/plain"):
        files = {"file": (filename, io.BytesIO(content), mime_type)}
        resp = client.post("/file/upload", files=files)
        return resp

    return _upload


@pytest.fixture
def uploaded_test_file(client, sample_file):
    """Upload the sample test file and return its metadata."""
    filename, content = sample_file
    files = {"file": (filename, io.BytesIO(content), "text/plain")}
    resp = client.post("/file/upload", files=files)
    return resp.json()


@pytest.fixture
def large_file_content():
    """Return content that will create multiple chunks."""
    return b"x" * (VFGConfig.DEFAULT_CHUNK_SIZE * 3)


@pytest.fixture
def get_metrics(client):
    """Factory fixture to get current metrics."""

    def _get_metrics():
        return client.get("/metrics").json()

    return _get_metrics


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
    assert len(resp.json()["filenames"]) == 0


def test_file_upload_pdf_returns_201(upload_file, sample_file):
    """Test that uploading a PDF file returns 201 status."""
    filename, content = sample_file
    resp = upload_file(filename, content)
    assert resp.status_code == 201


def test_file_upload_txt_returns_201(upload_file, sample_file):
    """Test that uploading a TXT file returns 201 status."""
    filename, content = sample_file
    resp = upload_file(filename, content)
    assert resp.status_code == 201


def test_file_upload_creates_multiple_chunks(upload_file, large_file_content):
    """Test that uploading a file creates multiple document chunks."""
    resp = upload_file("test_document.txt", large_file_content)
    assert resp.status_code == 201
    assert len(resp.json()["doc_ids"]) > 0


def test_file_upload_returns_chunk_count(uploaded_test_file):
    """Test that file upload response includes the number of chunks created."""
    assert "chunks_created" in uploaded_test_file
    assert isinstance(uploaded_test_file["chunks_created"], int)


def test_file_upload_returns_doc_ids(uploaded_test_file):
    """Test that file upload response includes all document ID strings."""
    assert "doc_ids" in uploaded_test_file
    assert isinstance(uploaded_test_file["doc_ids"], list)
    assert all(isinstance(doc_id, str) for doc_id in uploaded_test_file["doc_ids"])


def test_file_upload_chunks_have_metadata(client, uploaded_test_file):
    """Test that uploaded file chunks contain source_file and chunk_index metadata."""
    for doc_id in uploaded_test_file["doc_ids"]:
        resp = client.get(f"/doc/{doc_id}")
        assert resp.status_code == 200

        doc_data = resp.json()
        assert "metadata" in doc_data
        assert "source_file" in doc_data["metadata"]
        assert isinstance(doc_data["metadata"]["source_file"], str)
        assert "chunk_index" in doc_data["metadata"]
        assert isinstance(doc_data["metadata"]["chunk_index"], int)


def test_file_upload_unsupported_format_returns_400(upload_file):
    """Test that uploading an unsupported file format returns 400."""
    resp = upload_file("test_document.md", b"Test content", "text/unsupported-filetype")
    assert resp.status_code == 400


def test_file_upload_no_filename_returns_400(client):
    """Test that uploading without a filename returns 400."""
    files = {"file": (None, io.BytesIO(b"Test content"), "text/plain")}
    resp = client.post("/file/upload", files=files)
    assert resp.status_code in [400, 422]


def test_file_upload_empty_file(upload_file):
    """Test uploading an empty file."""
    resp = upload_file("test_document.txt", b"")
    assert resp.status_code == 400


def test_file_delete_returns_200(client, uploaded_test_file):
    """Test that file DELETE returns 200 for existing file."""
    resp = client.delete(f"/file/delete/{uploaded_test_file['filename']}")
    assert resp.status_code == 200


def test_file_delete_removes_all_chunks(client, uploaded_test_file):
    """Test that deleting a file removes all associated chunks."""
    filename = uploaded_test_file["filename"]
    resp = client.delete(f"/file/delete/{filename}")
    assert resp.status_code == 200

    for doc_id in resp.json()["doc_ids"]:
        resp = client.get(f"/doc/{doc_id}")
        assert resp.status_code == 404


def test_file_delete_returns_404_for_nonexistent_file(client):
    """Test that deleting a non-existent file returns 404."""
    resp = client.delete("/file/delete/nonexistant-file.txt")
    assert resp.status_code == 404


def test_file_delete_response_contains_filename(client, uploaded_test_file):
    """Test that delete response includes the filename."""
    resp = client.delete(f"/file/delete/{uploaded_test_file['filename']}")
    assert resp.status_code == 200

    data = resp.json()
    assert "filename" in data
    assert isinstance(data["filename"], str)


def test_file_delete_response_contains_chunks_deleted_count(client, uploaded_test_file):
    """Test that delete response includes count of chunks deleted."""
    resp = client.delete(f"/file/delete/{uploaded_test_file['filename']}")
    assert resp.status_code == 200

    data = resp.json()
    assert "chunks_deleted" in data
    assert isinstance(data["chunks_deleted"], int)


def test_file_delete_response_contains_doc_ids(client, uploaded_test_file):
    """Test that delete response includes list of deleted document IDs."""
    resp = client.delete(f"/file/delete/{uploaded_test_file['filename']}")
    assert resp.status_code == 200

    data = resp.json()
    assert "doc_ids" in data
    assert isinstance(data["doc_ids"], list)
    assert all(isinstance(doc_id, str) for doc_id in data["doc_ids"])


def test_file_list_includes_uploaded_filenames(client, uploaded_test_file):
    """Test that file list includes filenames of uploaded files."""
    resp = client.get("/file/list")
    assert resp.status_code == 200
    assert uploaded_test_file["filename"] in resp.json()["filenames"]


def test_file_list_excludes_deleted_files(client, uploaded_test_file):
    """Test that file list doesn't include deleted files."""
    filename = uploaded_test_file["filename"]

    resp = client.delete(f"/file/delete/{filename}")
    assert resp.status_code == 200

    resp = client.get("/file/list")
    assert resp.status_code == 200
    assert filename not in resp.json()["filenames"]


def test_file_upload_with_duplicate_filename(upload_file):
    """Test uploading a file with the same filename as an existing file."""
    filename = "duplicate_test.txt"

    resp1 = upload_file(filename, b"First version")
    assert resp1.status_code == 201
    doc_ids_1 = resp1.json()["doc_ids"]

    resp2 = upload_file(filename, b"Second version with different content")
    assert resp2.status_code == 201
    doc_ids_2 = resp2.json()["doc_ids"]

    assert len(doc_ids_1) > 0
    assert len(doc_ids_2) > 0
    assert set(doc_ids_1).isdisjoint(set(doc_ids_2))


def test_file_delete_does_not_affect_other_files(client, upload_file):
    """Test that deleting one file doesn't affect other uploaded files."""
    resp1 = upload_file("file1.txt", b"Content of first file")
    assert resp1.status_code == 201
    doc_ids_1 = resp1.json()["doc_ids"]

    resp2 = upload_file("file2.txt", b"Content of second file")
    assert resp2.status_code == 201
    doc_ids_2 = resp2.json()["doc_ids"]

    resp = client.delete("/file/delete/file1.txt")
    assert resp.status_code == 200

    for doc_id in doc_ids_1:
        assert client.get(f"/doc/{doc_id}").status_code == 404

    for doc_id in doc_ids_2:
        assert client.get(f"/doc/{doc_id}").status_code == 200


def test_file_upload_response_format(upload_file):
    """Test that file upload response contains all required fields."""
    filename = "format_test.txt"
    resp = upload_file(filename, b"Testing response format")
    assert resp.status_code == 201

    data = resp.json()
    assert "filename" in data
    assert "chunks_created" in data
    assert "doc_ids" in data
    assert "status" in data

    assert isinstance(data["filename"], str)
    assert isinstance(data["chunks_created"], int)
    assert isinstance(data["doc_ids"], list)
    assert isinstance(data["status"], str)

    assert data["filename"] == filename
    assert data["chunks_created"] == len(data["doc_ids"])
    assert data["status"] == "indexed"


def test_file_upload_large_pdf(upload_file):
    """Test uploading a large file that creates multiple chunks."""
    large_content = b"a" * (VFGConfig.DEFAULT_CHUNK_SIZE * 5)
    resp = upload_file("large_document.txt", large_content)

    assert resp.status_code == 201
    data = resp.json()
    assert data["chunks_created"] > 1
    assert len(data["doc_ids"]) > 1


def test_file_upload_pdf_with_special_characters_in_filename(upload_file):
    """Test uploading a file with special characters in filename."""
    filename = "test file with spaces & symbols (1).txt"
    resp = upload_file(filename, b"Content with special filename")

    assert resp.status_code == 201
    data = resp.json()
    assert data["filename"] == filename
    assert data["status"] == "indexed"


def test_file_delete_with_special_characters_in_filename(client, upload_file):
    """Test deleting a file with special characters in filename."""
    filename = "test file with spaces & symbols (1).txt"

    resp = upload_file(filename, b"Content for deletion")
    assert resp.status_code == 201

    resp = client.delete(f"/file/delete/{filename}")
    assert resp.status_code == 200

    data = resp.json()
    assert data["filename"] == filename
    assert data["status"] == "deleted"


def test_file_upload_pdf_extracts_text(client, upload_file):
    """Test that file upload correctly extracts text content."""
    resp = upload_file("extract_test.txt", b"Text content for extraction testing")

    assert resp.status_code == 201
    data = resp.json()
    assert data["chunks_created"] > 0
    assert len(data["doc_ids"]) > 0

    doc_resp = client.get(f"/doc/{data['doc_ids'][0]}")
    assert doc_resp.status_code == 200
    assert "content" in doc_resp.json()


def test_file_upload_txt_decodes_utf8(client, upload_file):
    """Test that TXT upload correctly decodes UTF-8 content."""
    utf8_content = "Hello 世界! Ça va? Здравствуй! 🎉"

    resp = upload_file("utf8_test.txt", utf8_content.encode("utf-8"))
    assert resp.status_code == 201

    doc_id = resp.json()["doc_ids"][0]
    doc_resp = client.get(f"/doc/{doc_id}")
    assert doc_resp.status_code == 200

    doc_data = doc_resp.json()
    assert "Hello" in doc_data["content"]
    assert utf8_content in doc_data["content"]


def test_file_upload_increments_files_uploaded_metric(get_metrics, upload_file):
    """Test that file upload increments files_uploaded metric."""
    initial_files = get_metrics()["usage"]["files_uploaded"]

    resp = upload_file("metric_test.txt", b"Testing file upload metric")
    assert resp.status_code == 201

    final_files = get_metrics()["usage"]["files_uploaded"]
    assert final_files == initial_files + 1


def test_file_upload_increments_chunks_created_metric(
    get_metrics, upload_file, large_file_content
):
    """Test that file upload increments chunks_created metric."""
    initial_chunks = get_metrics()["usage"]["chunks_created"]

    resp = upload_file("chunks_metric_test.txt", large_file_content)
    assert resp.status_code == 201
    chunks_created = resp.json()["chunks_created"]

    final_chunks = get_metrics()["usage"]["chunks_created"]
    assert final_chunks == initial_chunks + chunks_created
    assert chunks_created > 1


def test_file_delete_updates_deleted_docs_metric(client, get_metrics, upload_file):
    """Test that file deletion updates docs_deleted metric."""
    filename = "delete_metric_test.txt"
    upload_resp = upload_file(filename, b"Content for testing delete metrics")
    assert upload_resp.status_code == 201
    chunks_created = upload_resp.json()["chunks_created"]

    initial_deleted = get_metrics()["usage"]["documents_deleted"]

    delete_resp = client.delete(f"/file/delete/{filename}")
    assert delete_resp.status_code == 200
    chunks_deleted = delete_resp.json()["chunks_deleted"]

    final_deleted = get_metrics()["usage"]["documents_deleted"]
    assert final_deleted == initial_deleted + chunks_deleted
    assert chunks_deleted == chunks_created


def test_file_upload_response_contains_status(upload_file):
    """Test that upload response contains 'status' field."""
    resp = upload_file("status_test.txt", b"Testing status field")
    assert resp.status_code == 201

    data = resp.json()
    assert "status" in data
    assert data["status"] == "indexed"


def test_file_upload_response_contains_filename(upload_file):
    """Test that upload response contains filename."""
    filename = "filename_test.txt"
    resp = upload_file(filename, b"Testing filename field")
    assert resp.status_code == 201

    data = resp.json()
    assert "filename" in data
    assert data["filename"] == filename
    assert isinstance(data["filename"], str)


def test_file_delete_response_contains_status(client, upload_file):
    """Test that delete response contains 'status' field."""
    filename = "delete_status_test.txt"
    upload_resp = upload_file(filename, b"Testing delete status field")
    assert upload_resp.status_code == 201

    resp = client.delete(f"/file/delete/{filename}")
    assert resp.status_code == 200

    data = resp.json()
    assert "status" in data
    assert data["status"] == "deleted"


def test_file_upload_with_case_variant_extension(upload_file):
    """Test uploading files with .PDF or .TXT (uppercase) extension."""
    resp = upload_file("uppercase_test.TXT", b"Testing uppercase extension")
    assert resp.status_code == 400

    resp = upload_file(
        "uppercase_test.PDF", b"Testing uppercase extension", "application/pdf"
    )
    assert resp.status_code == 400


def test_file_list_after_multiple_uploads(client, upload_file):
    """Test file list after uploading multiple files."""
    filenames = ["file1.txt", "file2.txt", "file3.txt"]

    for filename in filenames:
        resp = upload_file(filename, f"Content of {filename}".encode("utf-8"))
        assert resp.status_code == 201

    resp = client.get("/file/list")
    assert resp.status_code == 200

    file_list = resp.json()["filenames"]
    for filename in filenames:
        assert filename in file_list


def test_file_upload_whitespace_only_content_returns_400(upload_file):
    """Test that uploading a file with only whitespace returns 400."""
    resp = upload_file("whitespace_only.txt", b"   \n\n\t\t   ")
    assert resp.status_code == 400


def test_file_delete_twice_returns_404(client, upload_file):
    """Test that deleting the same file twice returns 404 on second attempt."""
    filename = "double_delete_test.txt"

    resp = upload_file(filename, b"Testing double deletion")
    assert resp.status_code == 201

    resp = client.delete(f"/file/delete/{filename}")
    assert resp.status_code == 200

    resp = client.delete(f"/file/delete/{filename}")
    assert resp.status_code == 404


def test_file_upload_updates_last_file_uploaded_timestamp(get_metrics, upload_file):
    """Test that file upload updates the last_file_uploaded_at timestamp."""
    initial_timestamp = get_metrics()["timestamps"]["last_file_uploaded_at"]

    resp = upload_file("timestamp_test.txt", b"Testing timestamp update")
    assert resp.status_code == 201

    final_timestamp = get_metrics()["timestamps"]["last_file_uploaded_at"]
    assert final_timestamp != initial_timestamp
    assert final_timestamp is not None


def test_file_upload_chunk_metadata_contains_chunk_index(
    client, upload_file, large_file_content
):
    """Test that each chunk has the correct chunk_index in metadata."""
    resp = upload_file("chunk_index_test.txt", large_file_content)
    assert resp.status_code == 201

    doc_ids = resp.json()["doc_ids"]
    assert len(doc_ids) > 1

    for i, doc_id in enumerate(doc_ids):
        doc_resp = client.get(f"/doc/{doc_id}")
        assert doc_resp.status_code == 200

        metadata = doc_resp.json()["metadata"]
        assert "chunk_index" in metadata
        assert metadata["chunk_index"] == i


def test_file_upload_chunk_metadata_contains_source_file(client, upload_file):
    """Test that each chunk has the source_file in metadata."""
    filename = "source_file_test.txt"
    resp = upload_file(filename, b"Testing source file metadata")
    assert resp.status_code == 201

    for doc_id in resp.json()["doc_ids"]:
        doc_resp = client.get(f"/doc/{doc_id}")
        assert doc_resp.status_code == 200

        metadata = doc_resp.json()["metadata"]
        assert "source_file" in metadata
        assert metadata["source_file"] == filename


def test_file_list_response_structure(client):
    """Test that file list response has the correct structure."""
    resp = client.get("/file/list")
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, dict)
    assert "filenames" in data
    assert isinstance(data["filenames"], list)


def test_file_delete_response_structure(client, upload_file):
    """Test that file delete response has the correct structure."""
    filename = "structure_test.txt"
    resp = upload_file(filename, b"Testing delete response structure")
    assert resp.status_code == 201

    resp = client.delete(f"/file/delete/{filename}")
    assert resp.status_code == 200

    data = resp.json()
    assert isinstance(data, dict)
    assert "status" in data
    assert "filename" in data
    assert "chunks_deleted" in data
    assert "doc_ids" in data

    assert isinstance(data["status"], str)
    assert isinstance(data["filename"], str)
    assert isinstance(data["chunks_deleted"], int)
    assert isinstance(data["doc_ids"], list)


def test_file_upload_multiple_files_sequential(upload_file):
    """Test uploading multiple files sequentially."""
    for i in range(3):
        filename = f"sequential_{i}.txt"
        resp = upload_file(filename, f"Content of file {i}".encode("utf-8"))
        assert resp.status_code == 201
        assert resp.json()["filename"] == filename


def test_file_delete_empty_filename_returns_404(client):
    """Test that deleting with an empty filename returns 404."""
    resp = client.delete("/file/delete/")
    assert resp.status_code in [404, 405]


def test_file_upload_increases_total_documents(get_metrics, upload_file):
    """Test that file upload increases total_documents metric."""
    initial_docs = get_metrics()["index"]["total_documents"]

    resp = upload_file("docs_count_test.txt", b"Testing document count increase")
    assert resp.status_code == 201
    chunks_created = resp.json()["chunks_created"]

    final_docs = get_metrics()["index"]["total_documents"]
    assert final_docs == initial_docs + chunks_created


def test_file_upload_chunk_overlap_behavior(client, upload_file):
    """Test that chunks overlap correctly according to DEFAULT_CHUNK_OVERLAP."""
    chunk_size = VFGConfig.DEFAULT_CHUNK_SIZE
    content = "ABCDEFGH" * (chunk_size // 4)

    resp = upload_file("overlap_test.txt", content.encode("utf-8"))
    assert resp.status_code == 201

    doc_ids = resp.json()["doc_ids"]

    if len(doc_ids) > 1:
        resp1 = client.get(f"/doc/{doc_ids[0]}")
        resp2 = client.get(f"/doc/{doc_ids[1]}")

        chunk1 = resp1.json()["content"]
        chunk2 = resp2.json()["content"]

        assert len(chunk1) <= chunk_size + 10
        assert len(chunk2) <= chunk_size + 10


def test_file_delete_returns_all_chunk_ids_for_multipart_file(client, upload_file):
    """Test that deleting a multi-chunk file returns all chunk IDs."""
    large_content = b"x" * (VFGConfig.DEFAULT_CHUNK_SIZE * 4)

    upload_resp = upload_file("multipart_delete_test.txt", large_content)
    assert upload_resp.status_code == 201

    uploaded_doc_ids = set(upload_resp.json()["doc_ids"])
    chunks_created = upload_resp.json()["chunks_created"]

    delete_resp = client.delete("/file/delete/multipart_delete_test.txt")
    assert delete_resp.status_code == 200

    deleted_doc_ids = set(delete_resp.json()["doc_ids"])
    chunks_deleted = delete_resp.json()["chunks_deleted"]

    assert uploaded_doc_ids == deleted_doc_ids
    assert chunks_deleted == chunks_created
    assert chunks_deleted > 1


def test_file_upload_very_long_filename(upload_file):
    """Test uploading a file with a very long filename."""
    long_filename = "a" * 200 + ".txt"

    resp = upload_file(long_filename, b"Testing long filename")
    assert resp.status_code == 201
    assert resp.json()["filename"] == long_filename


def test_file_upload_filename_with_unicode_characters(client, upload_file):
    """Test uploading a file with Unicode characters in filename."""
    filename = "文档_тест_τεστ.txt"

    resp = upload_file(filename, b"Testing Unicode filename")
    assert resp.status_code == 201
    assert resp.json()["filename"] == filename

    list_resp = client.get("/file/list")
    assert filename in list_resp.json()["filenames"]

    delete_resp = client.delete(f"/file/delete/{filename}")
    assert delete_resp.status_code == 200


def test_file_delete_with_url_encoded_special_characters(client, upload_file):
    """Test deleting a file with URL-encoded special characters in path."""
    filename = "test file.txt"

    resp = upload_file(filename, b"Testing URL encoding")
    assert resp.status_code == 201

    resp = client.delete(f"/file/delete/{filename}")
    assert resp.status_code == 200


def test_file_list_returns_unique_filenames(client, upload_file):
    """Test that file list returns unique filenames even with duplicate uploads."""
    filename = "unique_test.txt"

    for i in range(3):
        resp = upload_file(filename, f"Version {i}".encode("utf-8"))
        assert resp.status_code == 201

    resp = client.get("/file/list")
    filenames = resp.json()["filenames"]

    assert filename in filenames
    filename_count = filenames.count(filename)
    assert filename_count >= 1


def test_file_upload_and_delete_affects_embeddings_count(
    get_metrics, upload_file, client
):
    """Test that upload increases and delete decreases embeddings count."""
    initial_metrics = get_metrics()
    initial_embeddings = initial_metrics["index"]["total_embeddings"]

    file_content = b"x" * (VFGConfig.DEFAULT_CHUNK_SIZE * 2)
    resp = upload_file("embeddings_test.txt", file_content)
    assert resp.status_code == 201
    chunks_created = resp.json()["chunks_created"]

    after_upload = get_metrics()
    assert (
        after_upload["index"]["total_embeddings"] == initial_embeddings + chunks_created
    )

    delete_resp = client.delete("/file/delete/embeddings_test.txt")
    assert delete_resp.status_code == 200

    after_delete = get_metrics()
    assert (
        after_delete["index"]["total_embeddings"]
        < after_upload["index"]["total_embeddings"]
    )
    assert after_delete["index"]["deleted_documents"] == 0


def test_file_upload_chunk_boundaries_no_content_loss(client, upload_file):
    """Test that chunking doesn't lose content at boundaries."""
    original_content = "A" * 100 + "B" * 200 + "C" * 300 + "D" * 400

    resp = upload_file("boundary_test.txt", original_content.encode("utf-8"))
    assert resp.status_code == 201

    doc_ids = resp.json()["doc_ids"]

    all_content = ""
    for doc_id in doc_ids:
        doc_resp = client.get(f"/doc/{doc_id}")
        chunk_content = doc_resp.json()["content"]

        if all_content and len(all_content) > 0:
            all_content += chunk_content
        else:
            all_content += chunk_content

    assert "A" * 100 in all_content
    assert "D" * 400 in all_content


def test_file_upload_single_character_file(client, upload_file):
    """Test uploading a file with minimal content (single character)."""
    resp = upload_file("single_char.txt", b"X")

    assert resp.status_code == 201
    data = resp.json()
    assert data["chunks_created"] == 1
    assert len(data["doc_ids"]) == 1

    doc_resp = client.get(f"/doc/{data['doc_ids'][0]}")
    assert doc_resp.json()["content"] == "X"


def test_file_upload_exact_chunk_size_boundary(upload_file):
    """Test uploading content that's exactly the chunk size."""
    file_content = b"X" * VFGConfig.DEFAULT_CHUNK_SIZE

    resp = upload_file("exact_size.txt", file_content)
    assert resp.status_code == 201
    assert resp.json()["chunks_created"] >= 1


def test_file_delete_mixed_files_preserves_others(client, upload_file):
    """Test deleting files in mixed order preserves correct documents."""
    files_data = [
        ("file_a.txt", b"Content A"),
        ("file_b.txt", b"Content B"),
        ("file_c.txt", b"Content C"),
    ]

    uploaded_files = {}
    for filename, content in files_data:
        resp = upload_file(filename, content)
        assert resp.status_code == 201
        uploaded_files[filename] = resp.json()["doc_ids"]

    resp = client.delete("/file/delete/file_b.txt")
    assert resp.status_code == 200

    for doc_id in uploaded_files["file_b.txt"]:
        assert client.get(f"/doc/{doc_id}").status_code == 404

    for doc_id in uploaded_files["file_a.txt"]:
        assert client.get(f"/doc/{doc_id}").status_code == 200
    for doc_id in uploaded_files["file_c.txt"]:
        assert client.get(f"/doc/{doc_id}").status_code == 200

    list_resp = client.get("/file/list")
    filenames = list_resp.json()["filenames"]
    assert "file_a.txt" in filenames
    assert "file_c.txt" in filenames
    assert "file_b.txt" not in filenames


def test_file_upload_newlines_and_formatting_preserved(client, upload_file):
    """Test that newlines and formatting in content are preserved."""
    content_with_formatting = "Line 1\n\nLine 2\n\tTabbed\n    Spaced"

    resp = upload_file("formatting_test.txt", content_with_formatting.encode("utf-8"))
    assert resp.status_code == 201

    doc_id = resp.json()["doc_ids"][0]
    doc_resp = client.get(f"/doc/{doc_id}")
    retrieved_content = doc_resp.json()["content"]

    assert "Line 1" in retrieved_content
    assert "Line 2" in retrieved_content
    assert "\n" in retrieved_content or "\\n" in retrieved_content


def test_file_operations_consistency_across_multiple_uploads_and_deletes(
    client, upload_file
):
    """Test system consistency with multiple interleaved upload/delete operations."""
    operations = [
        ("upload", "file1.txt", b"Content 1"),
        ("upload", "file2.txt", b"Content 2"),
        ("delete", "file1.txt", None),
        ("upload", "file3.txt", b"Content 3"),
        ("delete", "file2.txt", None),
        ("upload", "file4.txt", b"Content 4"),
    ]

    active_files = set()

    for op_type, filename, content in operations:
        if op_type == "upload":
            resp = upload_file(filename, content)
            assert resp.status_code == 201
            active_files.add(filename)
        else:
            resp = client.delete(f"/file/delete/{filename}")
            assert resp.status_code == 200
            active_files.discard(filename)

    list_resp = client.get("/file/list")
    current_files = set(list_resp.json()["filenames"])

    for active_file in active_files:
        assert active_file in current_files


def test_file_upload_exceeds_max_filename_length(upload_file):
    """Test that uploading with filename exceeding MAX_FILENAME_LENGTH returns 400."""
    long_filename = "a" * (VFGConfig.MAX_FILENAME_LENGTH + 1) + ".txt"

    resp = upload_file(long_filename, b"Testing filename too long")
    assert resp.status_code == 400
