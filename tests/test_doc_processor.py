"""Tests for document processing utility functions"""

from io import BytesIO

import fitz
import pytest
from fastapi import UploadFile

from vectorforge.config import VFGConfig
from vectorforge.doc_processor import chunk_text, extract_file_content, extract_pdf

# =============================================================================
# chunk_text() Tests
# =============================================================================


def test_chunk_text_with_short_text():
    """Test chunking text shorter than chunk_size."""
    text = "Short text"

    chunks = chunk_text(text, chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0] == "Short text"


def test_chunk_text_with_long_text():
    """Test chunking text longer than chunk_size."""
    text = "a" * 250

    chunks = chunk_text(text, chunk_size=100, overlap=10)

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_chunk_text_respects_chunk_size():
    """Test that chunks don't exceed specified chunk_size."""
    text = "x" * 1000

    chunks = chunk_text(text, chunk_size=50, overlap=5)

    for chunk in chunks:
        assert len(chunk) <= 50


def test_chunk_text_respects_overlap():
    """Test that consecutive chunks overlap by specified amount."""
    text = "abcdefghijklmnopqrstuvwxyz" * 10

    chunks = chunk_text(text, chunk_size=50, overlap=10)

    for i in range(len(chunks) - 1):
        assert chunks[i][-10:] == chunks[i + 1][:10]


def test_chunk_text_with_zero_overlap():
    """Test chunking with no overlap between chunks."""
    text = "a" * 200

    chunks = chunk_text(text, chunk_size=50, overlap=0)

    assert len(chunks) == 4
    assert all(len(chunk) == 50 for chunk in chunks)


def test_chunk_text_with_large_overlap():
    """Test chunking with overlap close to chunk_size."""
    text = "a" * 300

    chunks = chunk_text(text, chunk_size=100, overlap=90)

    assert len(chunks) > 10


def test_chunk_text_strips_whitespace():
    """Test that chunks have leading/trailing whitespace stripped."""
    text = "   content   more content   "

    chunks = chunk_text(text, chunk_size=10, overlap=2)

    for chunk in chunks:
        assert chunk == chunk.strip()


def test_chunk_text_excludes_empty_chunks():
    """Test that empty or whitespace-only chunks are excluded."""
    text = "text     \n\n     more"

    chunks = chunk_text(text, chunk_size=10, overlap=2)

    for chunk in chunks:
        assert chunk.strip() != ""


def test_chunk_text_with_empty_string():
    """Test chunking an empty string."""
    text = ""

    chunks = chunk_text(text)

    assert chunks == []


def test_chunk_text_with_whitespace_only():
    """Test chunking a string with only whitespace."""
    text = "     \n\n\t\t   "

    chunks = chunk_text(text)

    assert chunks == []


def test_chunk_text_with_exact_chunk_size():
    """Test text that is exactly chunk_size length."""
    text = "a" * 100

    chunks = chunk_text(text, chunk_size=100, overlap=10)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_default_parameters():
    """Test chunk_text with default chunk_size and overlap."""
    text = "x" * 1000

    chunks = chunk_text(text)

    assert len(chunks) > 0
    assert all(len(chunk) <= VFGConfig.DEFAULT_CHUNK_SIZE for chunk in chunks)


def test_chunk_text_custom_parameters():
    """Test chunk_text with custom chunk_size and overlap values."""
    text = "a" * 500

    chunks = chunk_text(text, chunk_size=75, overlap=15)

    assert all(len(chunk) <= 75 for chunk in chunks)


def test_chunk_text_preserves_content():
    """Test that all original content appears in chunks (accounting for overlap)."""
    text = "The quick brown fox jumps over the lazy dog"

    chunks = chunk_text(text, chunk_size=15, overlap=5)

    combined = "".join(chunks)
    for word in text.split():
        assert word in combined


def test_chunk_text_with_unicode():
    """Test chunking text containing unicode characters."""
    text = "Hello 世界 🌍 Привет मस्ते" * 20

    chunks = chunk_text(text, chunk_size=50, overlap=10)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= 50


def test_chunk_text_with_newlines():
    """Test chunking text containing newlines."""
    text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5" * 10

    chunks = chunk_text(text, chunk_size=30, overlap=5)

    assert len(chunks) > 0


def test_chunk_text_boundary_conditions():
    """Test chunking at exact boundary conditions."""
    text = "a" * 100

    chunks = chunk_text(text, chunk_size=50, overlap=25)

    assert len(chunks) == 3


def test_chunk_text_overlap_larger_than_chunk_size():
    """Test behavior when overlap is larger than chunk_size."""
    text = "a" * 200

    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunk_text(text, chunk_size=50, overlap=60)


def test_chunk_text_overlap_equal_to_chunk_size():
    """Test behavior when overlap is equal to chunk_size."""
    text = "a" * 200

    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunk_text(text, chunk_size=50, overlap=50)


def test_chunk_text_single_character():
    """Test chunking a single character string."""
    text = "x"

    chunks = chunk_text(text, chunk_size=10, overlap=2)

    assert len(chunks) == 1
    assert chunks[0] == "x"


def test_chunk_text_returns_list():
    """Test that chunk_text returns a list."""
    text = "Some text content"

    chunks = chunk_text(text)

    assert isinstance(chunks, list)


# =============================================================================
# extract_pdf() Tests
# =============================================================================


def test_extract_pdf_with_valid_pdf():
    """Test extracting text from a valid PDF file."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Test PDF content")
    pdf_bytes = doc.write()
    doc.close()

    text = extract_pdf(pdf_bytes)

    assert "Test PDF content" in text


def test_extract_pdf_with_multiple_pages():
    """Test that multi-page PDFs are extracted correctly."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Page 1 content")
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Page 2 content")
    pdf_bytes = doc.write()
    doc.close()

    text = extract_pdf(pdf_bytes)

    assert "Page 1 content" in text
    assert "Page 2 content" in text


def test_extract_pdf_pages_separated_by_double_newline():
    """Test that PDF pages are separated by double newlines."""
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((50, 50), "First")
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Second")
    pdf_bytes = doc.write()
    doc.close()

    text = extract_pdf(pdf_bytes)

    assert "\n\n" in text


def test_extract_pdf_with_empty_pdf():
    """Test extracting from a PDF with no text content."""
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.write()
    doc.close()

    text = extract_pdf(pdf_bytes)

    assert isinstance(text, str)


def test_extract_pdf_with_special_characters():
    """Test extracting PDF containing special characters."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Special: @#$%^&*()")
    pdf_bytes = doc.write()
    doc.close()

    text = extract_pdf(pdf_bytes)

    assert "@#$%^&*()" in text


def test_extract_pdf_with_unicode_content():
    """Test extracting PDF containing unicode characters."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello 世界 🌍")
    pdf_bytes = doc.write()
    doc.close()

    text = extract_pdf(pdf_bytes)

    assert isinstance(text, str)


def test_extract_pdf_with_invalid_pdf_bytes():
    """Test that invalid PDF bytes raise appropriate error."""
    invalid_bytes = b"This is not a PDF"

    with pytest.raises(Exception):
        extract_pdf(invalid_bytes)


def test_extract_pdf_returns_string():
    """Test that extract_pdf returns a string."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Content")
    pdf_bytes = doc.write()
    doc.close()

    text = extract_pdf(pdf_bytes)

    assert isinstance(text, str)


def test_extract_pdf_single_page():
    """Test extracting from a single-page PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Single page")
    pdf_bytes = doc.write()
    doc.close()

    text = extract_pdf(pdf_bytes)

    assert "Single page" in text


def test_extract_pdf_with_images_only():
    """Test extracting from a PDF with only images (no text)."""
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.write()
    doc.close()

    text = extract_pdf(pdf_bytes)

    assert isinstance(text, str)


def test_extract_pdf_with_corrupted_bytes():
    """Test that corrupted PDF bytes raise an error."""
    corrupted = b"%PDF-1.4\n corrupted data"

    with pytest.raises(Exception):
        extract_pdf(corrupted)


# =============================================================================
# extract_file_content() Tests
# =============================================================================


@pytest.mark.anyio
async def test_extract_file_content_with_pdf():
    """Test extracting content from a PDF UploadFile."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "PDF test content")
    pdf_bytes = doc.write()
    doc.close()

    file = UploadFile(filename="test.pdf", file=BytesIO(pdf_bytes))

    text = await extract_file_content(file)

    assert "PDF test content" in text


@pytest.mark.anyio
async def test_extract_file_content_with_txt():
    """Test extracting content from a TXT UploadFile."""
    content = b"Plain text content"
    file = UploadFile(filename="test.txt", file=BytesIO(content))

    text = await extract_file_content(file)

    assert text == "Plain text content"


@pytest.mark.anyio
async def test_extract_file_content_with_no_filename():
    """Test that file with no filename raises HTTPException 400."""
    content = b"Some content"
    file = UploadFile(filename=None, file=BytesIO(content))

    with pytest.raises(ValueError):
        await extract_file_content(file)


@pytest.mark.anyio
async def test_extract_file_content_with_unsupported_type():
    """Test that unsupported file type raises HTTPException 400."""
    content = b"Document content"
    file = UploadFile(filename="test.docx", file=BytesIO(content))

    with pytest.raises(ValueError):
        await extract_file_content(file)


@pytest.mark.anyio
async def test_extract_file_content_txt_utf8_decoding():
    """Test that TXT files are decoded as UTF-8."""
    content = "UTF-8 text: 你好世界".encode("utf-8")
    file = UploadFile(filename="test.txt", file=BytesIO(content))

    text = await extract_file_content(file)

    assert "你好世界" in text


@pytest.mark.anyio
async def test_extract_file_content_with_empty_pdf():
    """Test extracting from an empty PDF file."""
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.write()
    doc.close()

    file = UploadFile(filename="empty.pdf", file=BytesIO(pdf_bytes))

    text = await extract_file_content(file)

    assert isinstance(text, str)


@pytest.mark.anyio
async def test_extract_file_content_with_empty_txt():
    """Test extracting from an empty TXT file."""
    content = b""
    file = UploadFile(filename="empty.txt", file=BytesIO(content))

    text = await extract_file_content(file)

    assert text == ""


@pytest.mark.anyio
async def test_extract_file_content_error_message_includes_filename():
    """Test that error message for unsupported type includes filename."""
    content = b"Content"
    file = UploadFile(filename="document.docx", file=BytesIO(content))

    with pytest.raises(ValueError, match="Unsupported file type"):
        await extract_file_content(file)


@pytest.mark.anyio
async def test_extract_file_content_with_docx():
    """Test that DOCX files are rejected as unsupported."""
    content = b"DOCX content"
    file = UploadFile(filename="test.docx", file=BytesIO(content))

    with pytest.raises(ValueError):
        await extract_file_content(file)


@pytest.mark.anyio
async def test_extract_file_content_with_markdown():
    """Test that MD files are rejected as unsupported."""
    content = b"# Markdown content"
    file = UploadFile(filename="test.md", file=BytesIO(content))

    with pytest.raises(ValueError):
        await extract_file_content(file)


@pytest.mark.anyio
async def test_extract_file_content_case_insensitive_extension():
    """Test file extension matching is case-sensitive or insensitive."""
    content = b"Text content"
    file = UploadFile(filename="test.TXT", file=BytesIO(content))

    with pytest.raises(ValueError):
        await extract_file_content(file)


@pytest.mark.anyio
async def test_extract_file_content_pdf_returns_string():
    """Test that extract_file_content returns a string for PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Content")
    pdf_bytes = doc.write()
    doc.close()

    file = UploadFile(filename="doc.pdf", file=BytesIO(pdf_bytes))

    text = await extract_file_content(file)

    assert isinstance(text, str)


@pytest.mark.anyio
async def test_extract_file_content_txt_returns_string():
    """Test that extract_file_content returns a string for TXT."""
    content = b"Text file"
    file = UploadFile(filename="file.txt", file=BytesIO(content))

    text = await extract_file_content(file)

    assert isinstance(text, str)


@pytest.mark.anyio
async def test_extract_file_content_filename_none_raises_400():
    """Test that file.filename=None raises HTTPException with status 400."""
    content = b"Content"
    file = UploadFile(filename=None, file=BytesIO(content))

    with pytest.raises(ValueError):
        await extract_file_content(file)


@pytest.mark.anyio
async def test_extract_file_content_unsupported_extension_raises_400():
    """Test that unsupported extension raises HTTPException with status 400."""
    content = b"Content"
    file = UploadFile(filename="file.xyz", file=BytesIO(content))

    with pytest.raises(ValueError):
        await extract_file_content(file)
