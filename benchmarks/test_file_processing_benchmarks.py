"""File processing benchmarks for VectorForge.

Two tests covering the core file processing operations:
- Text chunking throughput (medium text)
- PDF extraction speed (medium synthetic PDF)

Medium input sizes are used throughout to reflect realistic inputs a typical query pipeline would encounter.
"""

import fitz
from faker import Faker

from vectorforge.doc_processor import chunk_text, extract_pdf


def test_chunk_medium_text(benchmark, sample_text_medium: str):
    """Benchmark chunking medium text (~1,000 words) at default settings."""
    benchmark(chunk_text, text=sample_text_medium, chunk_size=500, overlap=50)


def test_extract_pdf_synthetic_medium(benchmark):
    """Benchmark PDF extraction on a medium synthetic PDF (5 pages)."""
    fake = Faker()
    doc = fitz.open()

    for _ in range(5):
        page = doc.new_page()
        text_content = " ".join([fake.sentence() for _ in range(100)])
        page.insert_text((72, 72), text_content)

    pdf_bytes = doc.tobytes()
    doc.close()

    benchmark(extract_pdf, content=pdf_bytes)
