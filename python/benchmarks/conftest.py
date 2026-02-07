"""Pytest configuration and fixtures for VectorForge benchmarks.

Provides fixtures for:
- Test data generation at various scales
- Pre-populated VectorEngine instances
- Sample files for file processing tests
- Memory tracking utilities
"""

import tempfile
from typing import Any, Generator

import pytest
from faker import Faker

from vectorforge.vector_engine import VectorEngine

fake = Faker()
Faker.seed(42)


# ============================================================================
# Test Data Scales
# ============================================================================

SCALES = {
    "tiny": 10,
    "small": 100,
    "medium": 1_000,
    "large": 10_000,
    "xlarge": 50_000,
}


# ============================================================================
# Data Generation Utilities
# ============================================================================


def generate_document(
    doc_id: int, min_words: int = 20, max_words: int = 200
) -> dict[str, Any]:
    """Generate a realistic document with content and metadata.

    Args:
        doc_id: Unique identifier for the document
        min_words: Minimum number of words in document content
        max_words: Maximum number of words in document content

    Returns:
        Dictionary with 'content' and 'metadata' keys
    """
    # Generate realistic content
    num_sentences = fake.random_int(min=3, max=15)
    content = " ".join([fake.sentence() for _ in range(num_sentences)])

    # Generate metadata
    metadata = {
        "doc_id": doc_id,
        "author": fake.name(),
        "category": fake.random_element(
            ["tech", "science", "business", "health", "education"]
        ),
        "created_at": fake.date_time_this_year().isoformat(),
        "tags": [fake.word() for _ in range(fake.random_int(min=1, max=5))],
    }

    return {"content": content, "metadata": metadata}


def generate_file_chunk(
    chunk_id: int, source_file: str, chunk_index: int
) -> dict[str, Any]:
    """Generate a document chunk from a file.

    Args:
        chunk_id: Unique identifier for the chunk
        source_file: Name of the source file
        chunk_index: Index of this chunk within the file

    Returns:
        Dictionary with 'content' and 'metadata' keys
    """
    # Generate realistic chunk content
    num_sentences = fake.random_int(min=5, max=10)
    content = " ".join([fake.sentence() for _ in range(num_sentences)])

    metadata = {
        "source_file": source_file,
        "chunk_index": chunk_index,
        "chunk_id": chunk_id,
    }

    return {"content": content, "metadata": metadata}


def generate_documents(count: int) -> list[dict[str, Any]]:
    """Generate a list of realistic documents.

    Args:
        count: Number of documents to generate

    Returns:
        List of document dictionaries
    """
    return [generate_document(i) for i in range(count)]


def generate_query(complexity: str = "simple") -> str:
    """Generate a search query.

    Args:
        complexity: Type of query - "simple", "medium", or "complex"

    Returns:
        Query string
    """
    if complexity == "simple":
        return fake.word()
    elif complexity == "medium":
        return " ".join([fake.word() for _ in range(3)])
    else:
        return fake.sentence()


# ============================================================================
# Engine Fixtures
# ============================================================================


@pytest.fixture
def empty_engine() -> Generator[VectorEngine, None, None]:
    """Provide a fresh, empty VectorEngine instance.

    Yields:
        Empty VectorEngine instance
    """
    engine = VectorEngine()
    yield engine
    # Cleanup - engine will be garbage collected


@pytest.fixture
def engine_tiny(empty_engine: VectorEngine) -> VectorEngine:
    """Provide a VectorEngine with tiny dataset (10 docs).

    Args:
        empty_engine: Empty VectorEngine instance

    Returns:
        VectorEngine with 10 documents indexed
    """
    docs = generate_documents(SCALES["tiny"])
    for doc in docs:
        empty_engine.add_doc(doc["content"], doc["metadata"])
    return empty_engine


@pytest.fixture
def engine_small(empty_engine: VectorEngine) -> VectorEngine:
    """Provide a VectorEngine with small dataset (100 docs).

    Args:
        empty_engine: Empty VectorEngine instance

    Returns:
        VectorEngine with 100 documents indexed
    """
    docs = generate_documents(SCALES["small"])
    for doc in docs:
        empty_engine.add_doc(doc["content"], doc["metadata"])
    return empty_engine


@pytest.fixture
def engine_medium(empty_engine: VectorEngine) -> VectorEngine:
    """Provide a VectorEngine with medium dataset (1,000 docs).

    Args:
        empty_engine: Empty VectorEngine instance

    Returns:
        VectorEngine with 1,000 documents indexed
    """
    docs = generate_documents(SCALES["medium"])
    for doc in docs:
        empty_engine.add_doc(doc["content"], doc["metadata"])
    return empty_engine


@pytest.fixture
def engine_large() -> Generator[VectorEngine, None, None]:
    """Provide a VectorEngine with large dataset (10,000 docs).

    Note: Created separately to avoid dependency on empty_engine
    for better memory management with large datasets.

    Yields:
        VectorEngine with 10,000 documents indexed
    """
    engine = VectorEngine()
    docs = generate_documents(SCALES["large"])
    for doc in docs:
        engine.add_doc(doc["content"], doc["metadata"])
    yield engine


@pytest.fixture
def engine_xlarge() -> Generator[VectorEngine, None, None]:
    """Provide a VectorEngine with extra-large dataset (50,000 docs).

    Note: This is a heavy fixture - use sparingly for scaling tests.

    Yields:
        VectorEngine with 50,000 documents indexed
    """
    engine = VectorEngine()
    docs = generate_documents(SCALES["xlarge"])
    for doc in docs:
        engine.add_doc(doc["content"], doc["metadata"])
    yield engine


# ============================================================================
# Query Fixtures
# ============================================================================


@pytest.fixture
def simple_queries() -> list[str]:
    """Provide a list of simple single-word queries.

    Returns:
        List of 10 simple queries
    """
    return [generate_query("simple") for _ in range(10)]


@pytest.fixture
def medium_queries() -> list[str]:
    """Provide a list of medium complexity queries.

    Returns:
        List of 10 medium queries
    """
    return [generate_query("medium") for _ in range(10)]


@pytest.fixture
def complex_queries() -> list[str]:
    """Provide a list of complex sentence queries.

    Returns:
        List of 10 complex queries
    """
    return [generate_query("complex") for _ in range(10)]


# ============================================================================
# File Processing Fixtures
# ============================================================================


@pytest.fixture
def sample_text_small() -> str:
    """Generate small text content (100-500 words).

    Returns:
        Text string
    """
    num_sentences = fake.random_int(min=20, max=50)
    return " ".join([fake.sentence() for _ in range(num_sentences)])


@pytest.fixture
def sample_text_medium() -> str:
    """Generate medium text content (500-2000 words).

    Returns:
        Text string
    """
    num_sentences = fake.random_int(min=100, max=300)
    return " ".join([fake.sentence() for _ in range(num_sentences)])


@pytest.fixture
def sample_text_large() -> str:
    """Generate large text content (2000-5000 words).

    Returns:
        Text string
    """
    num_sentences = fake.random_int(min=400, max=800)
    return " ".join([fake.sentence() for _ in range(num_sentences)])


# ============================================================================
# Persistence Fixtures
# ============================================================================


@pytest.fixture
def temp_save_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for save/load tests.

    Yields:
        Path to temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# Benchmark Configuration
# ============================================================================


def pytest_benchmark_update_json(
    config: Any, benchmarks: Any, output_json: dict[str, Any]
) -> None:
    """Add custom metadata to benchmark JSON output.

    Args:
        config: Pytest config object
        benchmarks: Benchmark results
        output_json: JSON output dictionary to update
    """
    output_json["vectorforge_version"] = "0.9.0"
    output_json["benchmark_scales"] = SCALES


def pytest_configure(config: Any) -> None:
    """Configure pytest for benchmarks.

    Args:
        config: Pytest config object
    """
    config.addinivalue_line(
        "markers",
        "scale(size): mark test with a specific scale (tiny, small, medium, large, xlarge)",
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")
