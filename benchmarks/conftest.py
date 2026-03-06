"""Pytest configuration and fixtures for VectorForge benchmarks.

Provides fixtures for:
- Test data generation at various scales
- Pre-populated VectorEngine instances backed by EphemeralClient (fast, no disk I/O)
- Pre-populated VectorEngine instances backed by PersistentClient (realistic, with disk)
- Sample files for file processing tests
- Memory tracking utilities

Design notes:
- ``shared_model`` is session-scoped: the SentenceTransformer is loaded exactly once
  per benchmark run (~5 s on first call) and reused across all fixtures.
- ``make_ephemeral_engine`` produces engines backed by an in-memory ChromaDB
  EphemeralClient — best for isolating algorithmic performance from disk I/O.
- ``make_persistent_engine`` produces engines backed by a PersistentClient in a
  temporary directory — best for persistence/cold-start/scaling benchmarks.
- Pre-populated scale fixtures (empty/tiny/small/medium/large/xlarge) use
  EphemeralClient to keep fixture setup time predictable.
- SharedSystemClient.clear_system_cache() is called in PersistentClient teardowns
  to prevent file-descriptor leaks across many benchmark tests.
- Fixture population uses ``_bulk_populate`` (batch encode + single collection.add)
  instead of ``engine.add_doc`` to avoid per-document SQLite writes during setup.
  This makes medium (1 k docs) and large (10 k docs) fixtures feasible to create.
"""

import tempfile
import uuid
from typing import Any, Callable, Generator

import chromadb
import numpy as np
import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from faker import Faker
from sentence_transformers import SentenceTransformer

from vectorforge.config import VFGConfig
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
    num_sentences = fake.random_int(min=3, max=15)
    content = " ".join([fake.sentence() for _ in range(num_sentences)])

    metadata = {
        "doc_id": doc_id,
        "author": fake.name(),
        "category": fake.random_element(
            ["tech", "science", "business", "health", "education"]
        ),
        "created_at": fake.date_time_this_year().isoformat(),
        "tags": " ".join([fake.word() for _ in range(fake.random_int(min=1, max=5))]),
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


def _bulk_populate(
    engine: VectorEngine, docs: list[dict[str, Any]], batch_size: int = 500
) -> None:
    """Populate a VectorEngine with documents using fast bulk insertion.

    Bypasses ``engine.add_doc`` (which does a SQLite write per document) and
    instead encodes all documents in a single batched ``model.encode`` call,
    then inserts them into the underlying ChromaDB collection directly.  This
    is appropriate for fixture setup where the goal is to reach a target index
    size quickly, not to benchmark the insertion path itself.

    Args:
        engine: VectorEngine instance to populate.
        docs: List of document dicts with 'content' and 'metadata' keys.
        batch_size: Number of documents per ``collection.add`` call.  Kept
            below ChromaDB's default max-batch-size (41,666) to avoid OOM on
            large fixtures; 500 is a safe default.
    """
    contents = [d["content"] for d in docs]
    metadatas = [d["metadata"] for d in docs]
    ids = [str(uuid.uuid4()) for _ in docs]

    embeddings: np.ndarray = engine.model.encode(
        contents, convert_to_numpy=True, show_progress_bar=False
    )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized: np.ndarray = embeddings / norms

    for start in range(0, len(docs), batch_size):
        end = min(start + batch_size, len(docs))
        engine.collection.add(
            ids=ids[start:end],
            embeddings=normalized[start:end].tolist(),
            documents=contents[start:end],
            metadatas=metadatas[start:end],
        )


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
# Shared Model (session-scoped — loaded once for the entire benchmark run)
# ============================================================================


@pytest.fixture(scope="session")
def shared_model() -> SentenceTransformer:
    """Load the SentenceTransformer model once for the entire benchmark session.

    Session-scoped to avoid the ~5 s model-load cost on every test. All engine
    fixtures receive a reference to this single model instance.

    Returns:
        Loaded SentenceTransformer model.
    """
    return SentenceTransformer(VFGConfig.MODEL_NAME)


# ============================================================================
# Engine Factory Helpers
# ============================================================================


@pytest.fixture
def make_ephemeral_engine(
    shared_model: SentenceTransformer,
) -> Callable[[], VectorEngine]:
    """Provide a factory that creates fresh in-memory VectorEngine instances.

    Uses ChromaDB's EphemeralClient so no disk I/O is involved.  Ideal for
    benchmarks that measure pure algorithmic performance (search latency,
    indexing throughput, etc.).

    Args:
        shared_model: Session-scoped SentenceTransformer model.

    Returns:
        Zero-argument callable that returns a new, empty VectorEngine each call.
    """

    def _factory() -> VectorEngine:
        client = chromadb.EphemeralClient()
        collection = client.get_or_create_collection(
            name=VFGConfig.DEFAULT_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return VectorEngine(
            collection=collection,
            model=shared_model,
            chroma_client=client,
        )

    return _factory


@pytest.fixture
def make_persistent_engine(
    shared_model: SentenceTransformer,
) -> Generator[Callable[[], tuple[VectorEngine, str]], None, None]:
    """Provide a factory that creates PersistentClient-backed VectorEngine instances.

    Each call to the returned factory creates a fresh engine in a new temporary
    directory. Callers receive both the engine and its persist path so they can
    inspect disk sizes, measure cold-start loads, etc.

    Clears the SharedSystemClient cache on teardown to release file descriptors.

    Args:
        shared_model: Session-scoped SentenceTransformer model.

    Yields:
        Zero-argument callable returning ``(VectorEngine, chroma_path)`` tuples.
    """
    tmpdirs: list[tempfile.TemporaryDirectory] = []

    def _factory() -> tuple[VectorEngine, str]:
        tmpdir = tempfile.TemporaryDirectory()
        tmpdirs.append(tmpdir)
        chroma_path = tmpdir.name
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_or_create_collection(
            name=VFGConfig.DEFAULT_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        return (
            VectorEngine(
                collection=collection,
                model=shared_model,
                chroma_client=client,
            ),
            chroma_path,
        )

    yield _factory

    SharedSystemClient.clear_system_cache()
    for d in tmpdirs:
        d.cleanup()


# ============================================================================
# Pre-populated Engine Fixtures (EphemeralClient)
# ============================================================================


@pytest.fixture
def empty_engine(make_ephemeral_engine: Callable[[], VectorEngine]) -> VectorEngine:
    """Provide a fresh, empty in-memory VectorEngine instance.

    Args:
        make_ephemeral_engine: Factory for in-memory engines.

    Returns:
        Empty VectorEngine instance.
    """
    return make_ephemeral_engine()


@pytest.fixture
def engine_tiny(
    make_ephemeral_engine: Callable[[], VectorEngine],
) -> VectorEngine:
    """Provide a VectorEngine with tiny dataset (10 docs).

    Args:
        make_ephemeral_engine: Factory for in-memory engines.

    Returns:
        VectorEngine with 10 documents indexed.
    """
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES["tiny"]))
    return engine


@pytest.fixture
def engine_small(
    make_ephemeral_engine: Callable[[], VectorEngine],
) -> VectorEngine:
    """Provide a VectorEngine with small dataset (100 docs).

    Args:
        make_ephemeral_engine: Factory for in-memory engines.

    Returns:
        VectorEngine with 100 documents indexed.
    """
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES["small"]))
    return engine


@pytest.fixture
def engine_medium(
    make_ephemeral_engine: Callable[[], VectorEngine],
) -> VectorEngine:
    """Provide a VectorEngine with medium dataset (1,000 docs).

    Args:
        make_ephemeral_engine: Factory for in-memory engines.

    Returns:
        VectorEngine with 1,000 documents indexed.
    """
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES["medium"]))
    return engine


@pytest.fixture
def engine_large(
    make_ephemeral_engine: Callable[[], VectorEngine],
) -> Generator[VectorEngine, None, None]:
    """Provide a VectorEngine with large dataset (10,000 docs).

    Note:
        Marked ``@slow`` in tests that use this fixture. Created as a
        generator to support any future cleanup.

    Args:
        make_ephemeral_engine: Factory for in-memory engines.

    Yields:
        VectorEngine with 10,000 documents indexed.
    """
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES["large"]))
    yield engine


@pytest.fixture
def engine_xlarge(
    make_ephemeral_engine: Callable[[], VectorEngine],
) -> Generator[VectorEngine, None, None]:
    """Provide a VectorEngine with extra-large dataset (50,000 docs).

    Note:
        This is a heavy fixture — only use it in tests marked ``@slow``.

    Args:
        make_ephemeral_engine: Factory for in-memory engines.

    Yields:
        VectorEngine with 50,000 documents indexed.
    """
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES["xlarge"]))
    yield engine


# ============================================================================
# Query Fixtures
# ============================================================================


@pytest.fixture
def simple_queries() -> list[str]:
    """Provide a list of simple single-word queries.

    Returns:
        List of 10 simple queries.
    """
    return [generate_query("simple") for _ in range(10)]


@pytest.fixture
def medium_queries() -> list[str]:
    """Provide a list of medium complexity queries.

    Returns:
        List of 10 medium queries.
    """
    return [generate_query("medium") for _ in range(10)]


@pytest.fixture
def complex_queries() -> list[str]:
    """Provide a list of complex sentence queries.

    Returns:
        List of 10 complex queries.
    """
    return [generate_query("complex") for _ in range(10)]


# ============================================================================
# File Processing Fixtures
# ============================================================================


@pytest.fixture
def sample_text_small() -> str:
    """Generate small text content (100-500 words).

    Returns:
        Text string.
    """
    num_sentences = fake.random_int(min=20, max=50)
    return " ".join([fake.sentence() for _ in range(num_sentences)])


@pytest.fixture
def sample_text_medium() -> str:
    """Generate medium text content (500-2000 words).

    Returns:
        Text string.
    """
    num_sentences = fake.random_int(min=100, max=300)
    return " ".join([fake.sentence() for _ in range(num_sentences)])


@pytest.fixture
def sample_text_large() -> str:
    """Generate large text content (2000-5000 words).

    Returns:
        Text string.
    """
    num_sentences = fake.random_int(min=400, max=800)
    return " ".join([fake.sentence() for _ in range(num_sentences)])


# ============================================================================
# Persistence Fixtures
# ============================================================================


@pytest.fixture
def temp_save_dir() -> Generator[str, None, None]:
    """Provide a temporary directory for persistence tests.

    Yields:
        Path to temporary directory.
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
    output_json["vectorforge_version"] = "1.0.0"
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
