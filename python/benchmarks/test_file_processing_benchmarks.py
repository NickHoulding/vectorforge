"""File processing benchmarks for VectorForge.

Tests document processing performance:
- PDF text extraction
- Text file processing
- Text chunking at various sizes and overlaps
- End-to-end file upload workflow

Metrics tracked:
- Extraction time per file
- Extraction speed (KB/s, MB/s)
- Chunking speed (chunks/sec)
- Total processing time
"""

import fitz
from faker import Faker

from vectorforge.doc_processor import chunk_text, extract_pdf

# ============================================================================
# Text Chunking Benchmarks
# ============================================================================


def test_chunk_small_text(benchmark, sample_text_small: str):
    """Benchmark chunking small text (100-500 words)."""
    benchmark(chunk_text, text=sample_text_small, chunk_size=500, overlap=50)


def test_chunk_medium_text(benchmark, sample_text_medium: str):
    """Benchmark chunking medium text (500-2000 words)."""
    benchmark(chunk_text, text=sample_text_medium, chunk_size=500, overlap=50)


def test_chunk_large_text(benchmark, sample_text_large: str):
    """Benchmark chunking large text (2000-5000 words)."""
    benchmark(chunk_text, text=sample_text_large, chunk_size=500, overlap=50)


# ============================================================================
# Chunking - Varying Chunk Size
# ============================================================================


def test_chunk_size_200(benchmark, sample_text_medium: str):
    """Benchmark chunking with chunk_size=200."""
    benchmark(chunk_text, text=sample_text_medium, chunk_size=200, overlap=20)


def test_chunk_size_500(benchmark, sample_text_medium: str):
    """Benchmark chunking with chunk_size=500 (default)."""
    benchmark(chunk_text, text=sample_text_medium, chunk_size=500, overlap=50)


def test_chunk_size_1000(benchmark, sample_text_medium: str):
    """Benchmark chunking with chunk_size=1000."""
    benchmark(chunk_text, text=sample_text_medium, chunk_size=1000, overlap=100)


def test_chunk_size_2000(benchmark, sample_text_medium: str):
    """Benchmark chunking with chunk_size=2000."""
    benchmark(chunk_text, text=sample_text_medium, chunk_size=2000, overlap=200)


# ============================================================================
# Chunking - Varying Overlap
# ============================================================================


def test_chunk_overlap_0(benchmark, sample_text_medium: str):
    """Benchmark chunking with no overlap."""
    benchmark(chunk_text, text=sample_text_medium, chunk_size=500, overlap=0)


def test_chunk_overlap_25(benchmark, sample_text_medium: str):
    """Benchmark chunking with 25 char overlap."""
    benchmark(chunk_text, text=sample_text_medium, chunk_size=500, overlap=25)


def test_chunk_overlap_50(benchmark, sample_text_medium: str):
    """Benchmark chunking with 50 char overlap (default)."""
    benchmark(chunk_text, text=sample_text_medium, chunk_size=500, overlap=50)


def test_chunk_overlap_100(benchmark, sample_text_medium: str):
    """Benchmark chunking with 100 char overlap."""
    benchmark(chunk_text, text=sample_text_medium, chunk_size=500, overlap=100)


def test_chunk_overlap_200(benchmark, sample_text_medium: str):
    """Benchmark chunking with 200 char overlap."""
    benchmark(chunk_text, text=sample_text_medium, chunk_size=500, overlap=200)


# ============================================================================
# PDF Extraction Benchmarks
# ============================================================================


def test_extract_pdf_synthetic_small(benchmark):
    """Benchmark PDF extraction on small synthetic PDF."""
    fake = Faker()
    text_content = " ".join([fake.sentence() for _ in range(50)])

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text_content)
    pdf_bytes = doc.tobytes()
    doc.close()

    benchmark(extract_pdf, content=pdf_bytes)


def test_extract_pdf_synthetic_medium(benchmark):
    """Benchmark PDF extraction on medium synthetic PDF."""
    fake = Faker()
    doc = fitz.open()

    for _ in range(5):
        page = doc.new_page()
        text_content = " ".join([fake.sentence() for _ in range(100)])
        page.insert_text((72, 72), text_content)

    pdf_bytes = doc.tobytes()
    doc.close()

    benchmark(extract_pdf, content=pdf_bytes)


def test_extract_pdf_synthetic_large(benchmark):
    """Benchmark PDF extraction on large synthetic PDF."""
    fake = Faker()
    doc = fitz.open()

    for _ in range(20):
        page = doc.new_page()
        text_content = " ".join([fake.sentence() for _ in range(200)])
        page.insert_text((72, 72), text_content)

    pdf_bytes = doc.tobytes()
    doc.close()

    benchmark(extract_pdf, content=pdf_bytes)


# ============================================================================
# End-to-End File Processing
# ============================================================================


def test_e2e_process_text_file(benchmark, sample_text_medium: str, empty_engine):
    """Benchmark end-to-end text file processing and indexing."""

    def process_and_index():
        chunks = chunk_text(sample_text_medium, chunk_size=500, overlap=50)

        for i, chunk in enumerate(chunks):
            empty_engine.add_doc(
                content=chunk, metadata={"source_file": "test.txt", "chunk_index": i}
            )

    benchmark.pedantic(process_and_index, iterations=3, rounds=3)


def test_e2e_process_pdf_file(benchmark, empty_engine):
    """Benchmark end-to-end PDF processing and indexing."""
    fake = Faker()
    doc = fitz.open()

    for _ in range(10):
        page = doc.new_page()
        text_content = " ".join([fake.sentence() for _ in range(100)])
        page.insert_text((72, 72), text_content)

    pdf_bytes = doc.tobytes()
    doc.close()

    def process_and_index():
        text = extract_pdf(pdf_bytes)
        chunks = chunk_text(text, chunk_size=500, overlap=50)

        for i, chunk in enumerate(chunks):
            empty_engine.add_doc(
                content=chunk, metadata={"source_file": "test.pdf", "chunk_index": i}
            )

    benchmark.pedantic(process_and_index, iterations=2, rounds=2)


# ============================================================================
# Chunking Throughput
# ============================================================================


def test_chunking_throughput(benchmark, sample_text_large: str):
    """Measure chunking throughput (chunks per second)."""
    benchmark(chunk_text, text=sample_text_large, chunk_size=500, overlap=50)

    # Number of chunks will vary, but benchmark stats show time
    # Chunks per second = len(result) / time


# ============================================================================
# Memory Efficiency
# ============================================================================


def test_chunk_memory_efficiency(benchmark):
    """Test chunking on very large text to measure memory efficiency."""
    fake = Faker()
    large_text = " ".join([fake.sentence() for _ in range(100000)])

    def chunk_large():
        chunks = chunk_text(large_text, chunk_size=500, overlap=50)
        return len(chunks)

    benchmark.pedantic(chunk_large, iterations=1, rounds=3)


# ============================================================================
# Edge Cases
# ============================================================================


def test_chunk_exact_size(benchmark):
    """Benchmark chunking text that's exactly chunk_size."""
    text = "a" * 500
    benchmark(chunk_text, text=text, chunk_size=500, overlap=50)


def test_chunk_smaller_than_size(benchmark):
    """Benchmark chunking text smaller than chunk_size."""
    text = "a" * 100
    benchmark(chunk_text, text=text, chunk_size=500, overlap=50)


def test_chunk_no_overlap(benchmark, sample_text_medium: str):
    """Benchmark chunking with zero overlap."""
    benchmark(chunk_text, text=sample_text_medium, chunk_size=500, overlap=0)
