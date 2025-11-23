import fitz
from typing import cast


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
