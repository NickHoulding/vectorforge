"""Pytest configuration and fixtures for VectorForge benchmarks.

Provides:
- ``shared_model``: session-scoped SentenceTransformer, loaded once per run
- ``make_ephemeral_engine``: factory for fresh in-memory VectorEngine instances
- ``engine_small``, ``engine_medium``, ``engine_large``: pre-populated engines
- ``empty_engine``: empty in-memory engine
- ``sample_text_medium``: realistic text for file processing tests
- ``_bulk_populate``: fast bulk-insertion helper (bypasses per-doc SQLite writes)

Design notes:
- All scale fixtures use EphemeralClient (no disk I/O) to keep setup time low.
- ``_bulk_populate`` batch-encodes all documents in a single ``model.encode``
  call then inserts them via ``collection.add``: ~67x faster than ``add_doc``
  loops, making medium (1k docs) and large (10k docs) fixtures feasible.
- ``engine_large`` is only used by ``@pytest.mark.slow`` tests.
"""

import tempfile
import uuid
from typing import Any, Callable, Generator

import chromadb
import numpy as np
import pytest
from faker import Faker
from sentence_transformers import SentenceTransformer

from vectorforge.config import VFGConfig
from vectorforge.vector_engine import VectorEngine

fake = Faker()
Faker.seed(42)


# ============================================================================
# Session-level temp directory for MetricsStore
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def _metrics_temp_dir() -> Generator[None, None, None]:
    """Point VFGConfig.CHROMA_PERSIST_DIR at a temp directory for the session.

    VectorEngine derives the MetricsStore path from VFGConfig.CHROMA_PERSIST_DIR
    when an EphemeralClient is used (no persist_directory on the client settings).
    Without this fixture, MetricsStore would try to open a file inside a directory
    that may not exist, causing OperationalError.

    Yields:
        Nothing; restores the original value on teardown.
    """
    with tempfile.TemporaryDirectory() as tmp:
        original = VFGConfig.CHROMA_PERSIST_DIR
        VFGConfig.CHROMA_PERSIST_DIR = tmp
        yield
        VFGConfig.CHROMA_PERSIST_DIR = original


# ============================================================================
# Test Data Scales
# ============================================================================

SCALES = {
    "small": 100,
    "medium": 1_000,
    "large": 10_000,
}


# ============================================================================
# Data Generation Utilities
# ============================================================================


def generate_document(doc_id: int) -> dict[str, Any]:
    """Generate a realistic document with content and metadata.

    Args:
        doc_id: Unique identifier for the document.

    Returns:
        Dictionary with 'content' and 'metadata' keys.
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


def generate_documents(count: int) -> list[dict[str, Any]]:
    """Generate a list of realistic documents.

    Args:
        count: Number of documents to generate.

    Returns:
        List of document dictionaries.
    """
    return [generate_document(i) for i in range(count)]


def _bulk_populate(
    engine: VectorEngine, docs: list[dict[str, Any]], batch_size: int = 500
) -> None:
    """Populate a VectorEngine with documents using fast bulk insertion.

    Bypasses ``engine.add_doc`` (which triggers a SQLite write per document)
    by encoding all documents in one batched ``model.encode`` call and
    inserting them directly via ``collection.add``.  Use this for fixture
    setup only, not for benchmarking the insertion path itself.

    Args:
        engine: VectorEngine instance to populate.
        docs: List of document dicts with 'content' and 'metadata' keys.
        batch_size: Documents per ``collection.add`` call (default 500).
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


# ============================================================================
# Shared Model (session-scoped; loaded once for the entire benchmark run)
# ============================================================================


@pytest.fixture(scope="session")
def shared_model() -> SentenceTransformer:
    """Load the SentenceTransformer model once for the entire benchmark session.

    Returns:
        Loaded SentenceTransformer model.
    """
    return SentenceTransformer(VFGConfig.MODEL_NAME)


# ============================================================================
# Engine Factory and Pre-populated Fixtures
# ============================================================================


@pytest.fixture
def make_ephemeral_engine(
    shared_model: SentenceTransformer,
) -> Callable[[], VectorEngine]:
    """Provide a factory that creates fresh in-memory VectorEngine instances.

    Uses ChromaDB's EphemeralClient (no disk I/O involved).

    Args:
        shared_model: Session-scoped SentenceTransformer model.

    Returns:
        Zero-argument callable that returns a new empty VectorEngine each call.
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
def empty_engine(make_ephemeral_engine: Callable[[], VectorEngine]) -> VectorEngine:
    """Provide a fresh, empty in-memory VectorEngine.

    Returns:
        Empty VectorEngine instance.
    """
    return make_ephemeral_engine()


@pytest.fixture
def engine_small(
    make_ephemeral_engine: Callable[[], VectorEngine],
) -> VectorEngine:
    """Provide a VectorEngine pre-populated with 100 documents.

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
    """Provide a VectorEngine pre-populated with 1,000 documents.

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
    """Provide a VectorEngine pre-populated with 10,000 documents.

    Note:
        Only use in tests marked ``@pytest.mark.slow``.

    Yields:
        VectorEngine with 10,000 documents indexed.
    """
    engine = make_ephemeral_engine()
    _bulk_populate(engine, generate_documents(SCALES["large"]))
    yield engine


# ============================================================================
# File Processing Fixtures
# ============================================================================


@pytest.fixture
def sample_text_medium() -> str:
    """Generate medium-length text content (100–300 sentences).

    Returns:
        Text string suitable for chunking and PDF extraction benchmarks.
    """
    num_sentences = fake.random_int(min=100, max=300)
    return " ".join([fake.sentence() for _ in range(num_sentences)])


# ============================================================================
# Benchmark Configuration
# ============================================================================


def pytest_benchmark_update_json(
    config: Any, benchmarks: Any, output_json: dict[str, Any]
) -> None:
    """Add VectorForge metadata to benchmark JSON output.

    Args:
        config: Pytest config object.
        benchmarks: Benchmark results.
        output_json: JSON output dictionary to update.
    """
    output_json["vectorforge_version"] = "1.0.0"
    output_json["benchmark_scales"] = SCALES


def pytest_configure(config: Any) -> None:
    """Register custom markers.

    Args:
        config: Pytest config object.
    """
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow running (large-scale, skip with -m 'not slow')",
    )
