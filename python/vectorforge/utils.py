import fitz
from sentence_transformers import SentenceTransformer
from typing import cast, List, Dict, Any
from torch import Tensor


def extract_pdf(content: bytes) -> str:
    """Extract text from a PDF file"""
    with fitz.open(stream=content, filetype="pdf") as doc:
        pages: list[str] = [
            cast(str, page.get_text("text")) 
            for page in doc
        ]
        return "\n\n".join(pages)

def extract_file_content(content: bytes, file_type: str) -> str:
    """Extract the content of a file"""
    if file_type == 'txt':
        return content.decode('utf-8')
    elif file_type == '.pdf':
        return extract_pdf(content)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def create_chunks(
        text: str, 
        metadata: Dict[str, Any],
        chunk_size: int = 500, 
        overlap: int = 50
    ) -> List[Dict[str, Any]]:
    """Format text into chunks with metadata"""
    chunks: List[Dict[str, Any]] = []
    start: int = 0
    chunk_idx: int = 0

    while start < len(text):
        end: int = min(start + chunk_size, len(text))
        chunk_text: str = text[start:end].strip()

        if chunk_text:
            chunk: Dict[str, Any] = {
                "text": chunk_text,
                "chunk_index": chunk_idx,
                "metadata": metadata
            }
            chunks.append(chunk)

        start: int = end - overlap if end < len(text) else len(text)
        chunk_idx += 1

    return chunks

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Tensor]:
    """Generates embeddings for each chunk of text"""
    embeddings: List[Tensor] = []

    for chunk in chunks:
        embedding: Tensor = embedding_model.encode(chunk.get("text", ""))
        embeddings.append(embedding)

    return embeddings
