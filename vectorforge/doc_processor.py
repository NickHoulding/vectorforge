from typing import cast

import fitz
from fastapi import UploadFile

from vectorforge.config import VFGConfig


def extract_pdf(content: bytes) -> str:
    """Extract text content from a PDF file.

    Reads PDF content from a byte stream and extracts all text from each page,
    joining pages with double newlines for readability.

    Args:
        content: Raw bytes of the PDF file to process.

    Returns:
        Concatenated text content from all pages, with pages separated by
        double newlines (\n\n).

    Example:
        >>> pdf_bytes = open('document.pdf', 'rb').read()
        >>> text = extract_pdf(pdf_bytes)
        >>> print(text[:100])
    """
    with fitz.open(stream=content, filetype="pdf") as doc:
        pages: list[str] = [cast(str, page.get_text("text")) for page in doc]

        return "\n\n".join(pages)


async def extract_file_content(file: UploadFile) -> str:
    """Extract text content from an uploaded file.

    Processes uploaded files and extracts their text content based on file type.
    Currently supports PDF and plain text files. PDF extraction uses PyMuPDF (fitz)
    for accurate text extraction, while text files are decoded as UTF-8.

    Args:
        file: FastAPI UploadFile object containing the file to process.

    Returns:
        Extracted text content as a string. For PDFs, includes text from all
        pages concatenated together. For text files, returns the decoded content.

    Raises:
        ValueError: if an unsupported file type is provided.

    Example:
        >>> from fastapi import UploadFile
        >>> file = UploadFile(filename="document.pdf", file=...)
        >>> text = await extract_file_content(file)
    """
    if not file.filename:
        file.filename = ""

    content: bytes = await file.read()
    text: str = ""

    if file.filename.endswith(".pdf"):
        text = extract_pdf(content)
    elif file.filename.endswith(".txt"):
        text = content.decode("utf-8")
    else:
        raise ValueError(
            f"Unsupported file type: {file.filename}. Supported types: {VFGConfig.SUPPORTED_FILE_EXTENSIONS}"
        )

    return text


def chunk_text(
    text: str,
    chunk_size: int = VFGConfig.DEFAULT_CHUNK_SIZE,
    overlap: int = VFGConfig.DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks for semantic processing.

    Divides a long text document into smaller chunks with configurable size and
    overlap. Overlapping ensures context continuity across chunk boundaries,
    improving semantic search quality. Chunks are stripped of leading/trailing
    whitespace, and empty chunks are excluded.

    Args:
        text: Input text to split into chunks.
        chunk_size: Maximum number of characters per chunk. Defaults to 500.
        overlap: Number of characters to overlap between consecutive chunks.
            Defaults to 50. Overlap helps maintain context across boundaries.

    Returns:
        List of text chunks as strings, ordered sequentially from the original text.
        Empty or whitespace-only chunks are excluded from the result.

    Example:
        >>> text = "Long document content..." * 100
        >>> chunks = chunk_text(text, chunk_size=500, overlap=50)
        >>> len(chunks)
        20
        >>> # Each chunk overlaps by 50 characters with the next
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    chunks: list[str] = []
    start: int = 0

    while start < len(text):
        end: int = min(start + chunk_size, len(text))
        chunk: str = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap if end < len(text) else len(text)

    return chunks
