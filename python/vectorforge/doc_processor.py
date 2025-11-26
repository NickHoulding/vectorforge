import fitz
from fastapi import HTTPException, UploadFile
from typing import cast, List 

def extract_pdf(content: bytes) -> str:
    """Extract text from a PDF file"""
    with fitz.open(stream=content, filetype="pdf") as doc:
        pages: list[str] = [
            cast(str, page.get_text("text")) 
            for page in doc
        ]
        return "\n\n".join(pages)

async def extract_file_content(file: UploadFile) -> str:
    """Extract the text content of a file"""
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename provided"
        )
    
    content: bytes = await file.read()

    if file.filename.endswith('.pdf'):
        text: str = extract_pdf(content)
    elif file.filename.endswith('.txt'):
        text: str = content.decode('utf-8')
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.filename}"
        )
    
    return text

def chunk_text(
        text: str, 
        chunk_size: int = 500, 
        overlap: int = 50
    ) -> List[str]:
    """Format text into text chunks"""
    chunks: List[str] = []
    start: int = 0

    while start < len(text):
        end: int = min(start + chunk_size, len(text))
        chunk_text: str = text[start:end].strip()

        if chunk_text:
            chunks.append(chunk_text)

        start: int = end - overlap if end < len(text) else len(text)

    return chunks
