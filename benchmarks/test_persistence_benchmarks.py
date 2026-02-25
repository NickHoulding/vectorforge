"""Persistence benchmarks for VectorForge.

Tests save/load performance:
- Save time vs index size
- Load time vs index size
- Disk I/O efficiency
- Compression effectiveness
- Build vs compact operations

Metrics tracked:
- Save/load time
- File sizes (metadata, embeddings, total)
- Compression ratios
- Throughput (MB/s)
"""

import os

import pytest

from benchmarks.conftest import generate_document, generate_documents
from vectorforge.vector_engine import VectorEngine

# ============================================================================
# Save Performance
# ============================================================================


def test_save_tiny_index(benchmark, engine_tiny: VectorEngine, temp_save_dir: str):
    """Benchmark saving tiny index (10 docs)."""
    save_path = os.path.join(temp_save_dir, "tiny")
    benchmark(engine_tiny.save, directory=save_path)


def test_save_small_index(benchmark, engine_small: VectorEngine, temp_save_dir: str):
    """Benchmark saving small index (100 docs)."""
    save_path = os.path.join(temp_save_dir, "small")
    benchmark(engine_small.save, directory=save_path)


def test_save_medium_index(benchmark, engine_medium: VectorEngine, temp_save_dir: str):
    """Benchmark saving medium index (1,000 docs)."""
    save_path = os.path.join(temp_save_dir, "medium")
    benchmark(engine_medium.save, directory=save_path)


@pytest.mark.slow
def test_save_large_index(benchmark, engine_large: VectorEngine, temp_save_dir: str):
    """Benchmark saving large index (10,000 docs)."""
    save_path = os.path.join(temp_save_dir, "large")
    benchmark(engine_large.save, directory=save_path)


@pytest.mark.slow
def test_save_xlarge_index(benchmark, engine_xlarge: VectorEngine, temp_save_dir: str):
    """Benchmark saving extra-large index (50,000 docs)."""
    save_path = os.path.join(temp_save_dir, "xlarge")
    benchmark.pedantic(
        engine_xlarge.save, kwargs={"directory": save_path}, iterations=1, rounds=3
    )


# ============================================================================
# Load Performance
# ============================================================================


def test_load_tiny_index(benchmark, engine_tiny: VectorEngine, temp_save_dir: str):
    """Benchmark loading tiny index (10 docs)."""
    save_path = os.path.join(temp_save_dir, "tiny")
    engine_tiny.save(save_path)

    def load_index():
        engine = VectorEngine()
        engine.load(save_path)

    benchmark(load_index)


def test_load_small_index(benchmark, engine_small: VectorEngine, temp_save_dir: str):
    """Benchmark loading small index (100 docs)."""
    save_path = os.path.join(temp_save_dir, "small")
    engine_small.save(save_path)

    def load_index():
        engine = VectorEngine()
        engine.load(save_path)

    benchmark(load_index)


def test_load_medium_index(benchmark, engine_medium: VectorEngine, temp_save_dir: str):
    """Benchmark loading medium index (1,000 docs)."""
    save_path = os.path.join(temp_save_dir, "medium")
    engine_medium.save(save_path)

    def load_index():
        engine = VectorEngine()
        engine.load(save_path)

    benchmark(load_index)


@pytest.mark.slow
def test_load_large_index(benchmark, engine_large: VectorEngine, temp_save_dir: str):
    """Benchmark loading large index (10,000 docs)."""
    save_path = os.path.join(temp_save_dir, "large")
    engine_large.save(save_path)

    def load_index():
        engine = VectorEngine()
        engine.load(save_path)

    benchmark(load_index)


@pytest.mark.slow
def test_load_xlarge_index(benchmark, engine_xlarge: VectorEngine, temp_save_dir: str):
    """Benchmark loading extra-large index (50,000 docs)."""
    save_path = os.path.join(temp_save_dir, "xlarge")
    engine_xlarge.save(save_path)

    def load_index():
        engine = VectorEngine()
        engine.load(save_path)

    benchmark.pedantic(load_index, iterations=1, rounds=3)


# ============================================================================
# Save + Load Round Trip
# ============================================================================


def test_roundtrip_small(benchmark, engine_small: VectorEngine, temp_save_dir: str):
    """Benchmark save + load round trip for small index."""
    save_path = os.path.join(temp_save_dir, "roundtrip_small")

    def roundtrip():
        engine_small.save(save_path)
        engine = VectorEngine()
        engine.load(save_path)

    benchmark.pedantic(roundtrip, iterations=3, rounds=3)


def test_roundtrip_medium(benchmark, engine_medium: VectorEngine, temp_save_dir: str):
    """Benchmark save + load round trip for medium index."""
    save_path = os.path.join(temp_save_dir, "roundtrip_medium")

    def roundtrip():
        engine_medium.save(save_path)
        engine = VectorEngine()
        engine.load(save_path)

    benchmark.pedantic(roundtrip, iterations=2, rounds=2)


# ============================================================================
# Compression Effectiveness
# ============================================================================


def test_save_compression_ratio(engine_medium: VectorEngine, temp_save_dir: str):
    """Test compression ratio for saved index (not a benchmark, just measurement)."""
    save_path = os.path.join(temp_save_dir, "compression_test")
    result = engine_medium.save(save_path)

    total_size_mb = result["total_size_mb"]
    num_docs = result["documents_saved"]
    estimated_uncompressed_mb = (num_docs * 1536) / (1024 * 1024)

    compression_ratio = (
        estimated_uncompressed_mb / total_size_mb if total_size_mb > 0 else 0
    )

    assert compression_ratio > 0, f"Compression ratio: {compression_ratio:.2f}x"


# ============================================================================
# Incremental Save Performance
# ============================================================================


def test_save_after_additions(
    benchmark, engine_small: VectorEngine, temp_save_dir: str
):
    """Benchmark save performance after adding documents."""
    save_path = os.path.join(temp_save_dir, "incremental")

    new_docs = generate_documents(50)
    for doc in new_docs:
        engine_small.add_doc(doc["content"], doc["metadata"])

    benchmark(engine_small.save, directory=save_path)


def test_save_after_deletions(
    benchmark, engine_small: VectorEngine, temp_save_dir: str
):
    """Benchmark save performance after deleting documents."""
    save_path = os.path.join(temp_save_dir, "after_deletions")

    doc_ids = list(engine_small.documents.keys())
    for doc_id in doc_ids[:30]:
        engine_small.delete_doc(doc_id)

    benchmark(engine_small.save, directory=save_path)


# ============================================================================
# Multiple Save Operations
# ============================================================================


def test_repeated_saves(benchmark, engine_small: VectorEngine, temp_save_dir: str):
    """Benchmark repeated save operations (simulating periodic checkpoints)."""
    save_path = os.path.join(temp_save_dir, "repeated")

    def save_checkpoint():
        doc = generate_document(9999)
        engine_small.add_doc(doc["content"], doc["metadata"])
        engine_small.save(save_path)

    benchmark.pedantic(save_checkpoint, iterations=5, rounds=3)


# ============================================================================
# Cold Load Performance
# ============================================================================


def test_cold_load_with_search(
    benchmark, engine_medium: VectorEngine, temp_save_dir: str
):
    """Benchmark cold load followed by immediate search."""
    save_path = os.path.join(temp_save_dir, "cold_load")
    engine_medium.save(save_path)

    def cold_load_and_search():
        engine = VectorEngine()
        engine.load(save_path)
        engine.search("test query", top_k=10)

    benchmark.pedantic(cold_load_and_search, iterations=3, rounds=3)


# ============================================================================
# Disk Space Efficiency
# ============================================================================


def test_disk_space_per_document(engine_medium: VectorEngine, temp_save_dir: str):
    """Measure disk space per document (not a benchmark, just measurement)."""
    save_path = os.path.join(temp_save_dir, "space_test")
    result = engine_medium.save(save_path)

    total_size_mb = result["total_size_mb"]
    num_docs = result["documents_saved"]

    kb_per_doc = (total_size_mb * 1024) / num_docs if num_docs > 0 else 0

    assert kb_per_doc > 0, f"Disk space per document: {kb_per_doc:.2f} KB"
