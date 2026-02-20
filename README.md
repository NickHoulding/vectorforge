# VectorForge

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121.2+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-0.9.0-orange.svg)

> A high-performance vector database with semantic search capabilities, powered by ChromaDB, built with FastAPI and sentence transformers.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [What is VectorForge?](#what-is-vectorforge)
- [Demo](#demo)
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Screenshots](#screenshots)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Development Process](#development-process)
- [Known Limitations](#known-limitations--future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Problem Statement

Traditional keyword-based search systems struggle to understand the semantic meaning of documents. Users need to use exact keywords or phrases to find relevant information, missing documents that contain semantically similar content but use different terminology.

VectorForge solves this by:
- Converting documents into high-dimensional vector embeddings that capture semantic meaning
- Enabling similarity-based search that finds conceptually related content regardless of exact wording
- Providing a lightweight, persistent vector database powered by ChromaDB
- Offering real-time indexing and search with comprehensive metrics tracking

---

## What is VectorForge?

VectorForge is a **production-ready vector database** designed for semantic search applications. It combines the simplicity of a REST API with the power of transformer-based embeddings and ChromaDB's persistent storage to enable intelligent document retrieval.

Unlike traditional databases, VectorForge:
- **Understands meaning** - Uses sentence transformers to encode semantic content
- **Persistent by default** - ChromaDB automatically persists all data to disk
- **Provides observability** - Comprehensive metrics tracking for performance monitoring
- **Handles documents smartly** - Automatic chunking, metadata preservation, and file processing

Perfect for building:
- Semantic search engines
- Document Q&A systems
- Knowledge base retrieval
- Content recommendation systems
- RAG (Retrieval-Augmented Generation) backends

---

## Demo

**Live Demo**: [TODO]

<!-- Add GIF/Video Preview Here -->
<!-- Recommended: Screen recording showing:
  1. Document upload via API
  2. Semantic search query
  3. Relevant results returned
  4. Metrics dashboard
-->

---

## Tech Stack

### **Core Technologies**
- **Python 3.11+** - Modern Python with type hints and performance improvements
- **FastAPI 0.121+** - High-performance async web framework
- **Uvicorn** - ASGI server for production deployment
- **ChromaDB 0.5+** - High-performance embeddable vector database

### **Machine Learning & NLP**
- **sentence-transformers 5.1+** - State-of-the-art sentence embeddings
  - Model: `all-MiniLM-L6-v2` (384-dimensional embeddings)
- **NumPy** - Efficient vector operations and similarity calculations

### **Document Processing**
- **PyMuPDF (fitz) 1.26+** - Fast PDF text extraction
- **python-multipart** - File upload handling

### **Data Validation & Serialization**
- **Pydantic** - Data validation using Python type annotations (via FastAPI)

### **Development & Testing**
- **pytest 8.0+** - Testing framework with comprehensive test suite
- **pytest-cov** - Code coverage reporting
- **isort 7.0+** - Import sorting and organization
- **httpx** - Async HTTP client for testing

### **Deployment**
- **Docker** (optional) - Containerization support
- **uvicorn** - Production ASGI server

---

## Key Features

### **Semantic Search**
- Vector similarity search using cosine distance
- Configurable top-k results
- **Metadata filtering** - Filter results by exact metadata field matching (AND logic)
- Real-time embedding generation
- Query performance tracking with percentile metrics (p50, p95, p99)

### **Document Management**
- Add individual documents with custom metadata
- Retrieve documents by ID
- Delete documents (ChromaDB handles deletion immediately)
- Automatic embedding generation and normalization
- Content length validation (max 10,000 characters)

### **File Processing**
- **PDF Support** - Text extraction from PDF documents
- **TXT Support** - UTF-8 text file processing
- **Automatic Chunking** - Configurable chunk size (default: 500 chars) with overlap (default: 50 chars)
- **Batch Processing** - Upload files and get multiple document chunks
- **File-level Deletion** - Remove all chunks from a specific file

### **Index Management**
- **Automatic Persistence** - ChromaDB automatically persists all changes to disk
- **Index Statistics** - Real-time metrics on document counts and health
- **Directory Management** - Save/load operations support custom directory paths

### **Comprehensive Metrics**
- **Performance Metrics** - Query times, averages, min/max, percentiles
- **Usage Metrics** - Documents added/deleted, files uploaded, chunks created
- **Memory Metrics** - Embeddings size, documents size, total memory usage
- **System Info** - Model name, embedding dimension, uptime, version
- **Timestamps** - Track last query, document addition, file upload

### **Production-Ready**
- RESTful API with OpenAPI documentation
- Comprehensive error handling with detailed HTTP status codes
- Health check endpoint for monitoring
- Version tracking across all operations
- Extensive test coverage (~100% for core endpoints)

---

## Performance

**Baseline:** Python implementation v0.9.0 (tag: `v0.9.0-python-baseline`)

VectorForge delivers sub-20ms search latency for 10K documents using pure Python and NumPy. Performance scales linearly with index size for the current brute-force cosine similarity implementation.

### **Search Latency (Linear Scan)**

Search performance across different index sizes (top_k=10, single query):

| Index Size | Mean Latency | Median Latency | Operations/sec |
|------------|--------------|----------------|----------------|
| 10 docs    | 5.3 ms       | 5.2 ms         | 188 ops/sec    |
| 100 docs   | 5.5 ms       | 5.4 ms         | 183 ops/sec    |
| 1,000 docs | 6.8 ms       | 6.8 ms         | 147 ops/sec    |
| 10,000 docs| 18.9 ms      | 18.8 ms        | 53 ops/sec     |

**Note:** The similarity increases from 10 to 100 docs primarily due to embedding generation overhead (fixed cost). The vector similarity calculation itself scales linearly.

### **Indexing Throughput**

Document addition performance:

| Operation | Mean Latency | Throughput |
|-----------|--------------|------------|
| Single document (empty index) | 5.6 ms | ~179 docs/sec |
| Batch of 1,000 documents | 2.2 sec total | ~450 docs/sec |

**Note:** Most indexing time (~85-90%) is spent in transformer model inference for embedding generation, not in vector storage operations.

### **Memory Footprint**

Per-document memory consumption (384-dimensional embeddings):

- **Embedding vector:** ~1.5 KB (384 floats × 4 bytes)
- **Document storage:** ~0.5-2 KB (content + metadata, varies)
- **Index overhead:** ~0.1 KB (mappings, tracking)
- **Total per document:** ~2-4 KB

Example: 10,000 documents ≈ 20-40 MB RAM

### **Profiling Insights: Optimization Hotspots**

CPU profiling (10,000 documents, 100 search queries) reveals:

**Hotspot #1: Embedding Generation (85% of total time)**
- `model.encode()`: Transformer inference for query/document embedding
- **C++ opportunity:** Limited (requires ONNX or custom GPU kernels)
- **Status:** Not targeted for initial C++ optimization

**Hotspot #2: Cosine Similarity Calculation (1.2s for 1M calls)**
- `vector_engine.py:740` - `np.dot()` for normalized vectors
- **C++ opportunity:** High (vectorized SIMD operations, batch processing)
- **Expected speedup:** 5-10x with C++/Eigen or similar
- **Status:** **Primary target for C++/pybind11 optimization**

**Hotspot #3: Result Sorting (0.08s for 1M calls)**
- `vector_engine.py:390` - Python list sort with lambda
- **C++ opportunity:** Medium (native sorting algorithms)
- **Expected speedup:** 2-3x
- **Status:** Secondary optimization target

### **Performance Characteristics**

**Current Algorithm:** Brute-force linear scan (O(n) per query)
- **Best for:** <10K documents, exact k-NN results required
- **Limitations:** Scales linearly, slow for >100K documents

**Planned Optimization (C++ + HNSW):**
- **Target:** >100x speedup for 100K+ documents
- **Method:** Hierarchical Navigable Small World graph index
- **Trade-off:** ~1-5% recall loss for massive speed gains
- **Expected:** <5ms search on 1M documents

### **Benchmark Reproducibility**

All benchmarks were run on:
- **Platform:** Linux x86_64
- **Python:** 3.11.14
- **NumPy:** 2.4.1
- **Model:** sentence-transformers/all-MiniLM-L6-v2
- **Hardware:** (varies by machine)

To reproduce benchmarks on your system:

```bash
cd python

# Run search latency benchmarks
uv run pytest benchmarks/test_search_benchmarks.py::test_search_latency_tiny \
    benchmarks/test_search_benchmarks.py::test_search_latency_small \
    benchmarks/test_search_benchmarks.py::test_search_latency_medium \
    benchmarks/test_search_benchmarks.py::test_search_latency_large \
    --benchmark-only --benchmark-columns=mean,median,ops

# Run all benchmarks and save baseline
uv run pytest benchmarks/ --benchmark-only --benchmark-save=my_baseline

# Compare against saved baseline
uv run pytest benchmarks/ --benchmark-only --benchmark-compare=my_baseline
```

Full baseline data: `python/benchmarks/results/0001_baseline.json` (113 benchmarks)

---

## Architecture

### **Design Patterns & Technical Decisions**

#### **1. ChromaDB Vector Store**
VectorForge uses ChromaDB as its persistent vector database backend:
- **Automatic Persistence** - All data automatically saved to disk via ChromaDB's PersistentClient
- **Efficient Storage** - ChromaDB uses SQLite for metadata and optimized storage for embeddings
- **HNSW Indexing** - Hierarchical Navigable Small World graphs for fast similarity search
- **Cosine Similarity** - Collection configured with cosine distance metric

**Rationale**: ChromaDB provides production-ready persistence, efficient indexing, and eliminates the need for custom index management.

#### **2. Immediate Deletion**
Documents are immediately removed from the ChromaDB collection:
```python
collection.delete(ids=[doc_id])  # Immediate removal
```

**Benefits**:
- Consistent state - no stale data
- Simplified implementation - no compaction needed
- ChromaDB handles internal cleanup and optimization

#### **3. Normalized Embeddings**
All embeddings are L2-normalized during creation:
```python
embedding = embedding / np.linalg.norm(embedding)
```

**Benefits**:
- Cosine similarity reduces to dot product during search operations (faster)
- Numerical stability in similarity calculations

#### **4. Modular Architecture**
```
vectorforge/
├── api.py              # FastAPI endpoints (presentation layer)
├── vector_engine.py    # Core vector operations (business logic)
├── doc_processor.py    # Document processing utilities
└── models/             # Pydantic models (data layer)
    ├── documents.py
    ├── files.py
    ├── search.py
    ├── index.py
    └── metrics.py
```

**Rationale**: Separation of concerns enables testing and maintainability.

#### **5. Comprehensive Metrics Tracking**
VectorForge uses a dataclass-based metrics system:
```python
@dataclass
class EngineMetrics:
    total_queries: int
    query_times: deque[float]  # Rolling window
    # ... 15+ tracked metrics
```

**Benefits**:
- Zero-overhead tracking (in-memory counters)
- Historical query performance analysis
- Production debugging and optimization

#### **6. Metadata Coupling Validation**
File-based documents require both `source_file` and `chunk_index`:
```python
if has_source != has_chunk_index:
    raise ValueError("Must contain both or neither")
```

**Rationale**: Maintains data integrity for file reconstruction.

---

## Screenshots

### **API Documentation (Swagger UI)**

**TODO**
<!-- Screenshot: http://localhost:3001/docs -->
<!-- Should show: All endpoints organized by category with request/response schemas -->

### **Search Results**

**TODO**
<!-- Screenshot: POST /search response -->
<!-- Should show: JSON response with query, results array, similarity scores, metadata -->

### **Metrics Dashboard**

**TODO**
<!-- Screenshot: GET /metrics response -->
<!-- Should show: Comprehensive metrics including performance, usage, memory, timestamps -->

### **File Upload Flow**

**TODO**
<!-- Screenshot: POST /file/upload response -->
<!-- Should show: File uploaded, chunks created, document IDs returned -->

---

## Getting Started

### **Prerequisites**

- **Python 3.11 or higher**
- **uv** package manager
- **Git** (for cloning repository)

### **Installation**

1. **Clone the repository**
```bash
git clone https://github.com/NickHoulding/vectorforge.git
cd vectorforge
```

2. **Install dependencies**

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
cd python
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

3. **Verify installation**
```bash
python -c "import vectorforge; print(vectorforge.__version__)"
# Should output: 0.9.0
```

### **How to Run**

#### **Start the API Server**
```bash
cd python
uv run vectorforge/api.py
```

The API will be available at:
- **Base URL**: http://localhost:3001
- **Interactive Docs**: http://localhost:3001/docs
- **OpenAPI Schema**: http://localhost:3001/openapi.json

#### **Run Tests**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=vectorforge --cov-report=html

# Run specific test file
pytest tests/test_doc_endpoints.py -v

# Run tests matching pattern
pytest -k "search" -v
```

#### **View Coverage Report**
```bash
pytest --cov=vectorforge --cov-report=html
xdg-open htmlcov/index.html  # Linux
open htmlcov/index.html      # macOS
```

### **Environment Variables**

VectorForge currently uses hardcoded defaults, but you can customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `3001` | Port for FastAPI server |
| `DEFAULT_DATA_DIR` | `./data` | Directory for index persistence |
| `MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `COMPACTION_THRESHOLD` | `0.25` | Deletion ratio triggering compaction |

---

## Usage

### **1. Add a Document**

```bash
curl -X POST http://localhost:3001/doc/add \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Machine learning is a subset of artificial intelligence",
    "metadata": {
      "source_file": "ml_intro.txt",
      "chunk_index": 0,
      "author": "John Doe"
    }
  }'
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "indexed"
}
```

### **2. Search Documents**

```bash
curl -X POST http://localhost:3001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "query": "What is AI?",
  "results": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "Machine learning is a subset of artificial intelligence",
      "score": 0.8234,
      "metadata": {
        "source_file": "ml_intro.txt",
        "chunk_index": 0,
        "author": "John Doe"
      }
    }
  ],
  "count": 1
}
```

### **3. Search with Metadata Filters**

Filter search results by metadata fields using exact equality matching. All filters use AND logic (all must match).

**Key Features:**
- **Exact matching**: Filters use case-sensitive equality comparison
- **AND logic**: Multiple filters require all to match
- **Flexible filtering**: Filter by any metadata field independently
- **No pairing required**: `source_file` and `chunk_index` can be used separately or together
- **Performance**: Filtering happens after similarity scoring

#### **Common Filtering Scenarios:**

**1. Filter by source file (all chunks from a specific file):**

```bash
curl -X POST http://localhost:3001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning concepts",
    "top_k": 10,
    "filters": {
      "source_file": "textbook.pdf"
    }
  }'
```

Returns all matching chunks from `textbook.pdf`, regardless of chunk index.

**2. Filter by chunk index (first chunks from all files):**

```bash
curl -X POST http://localhost:3001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "introduction",
    "top_k": 10,
    "filters": {
      "chunk_index": 0
    }
  }'
```

Returns all matching first chunks (index 0) from any file. Useful for finding document introductions.

**3. Filter by both (specific chunk from specific file):**

```bash
curl -X POST http://localhost:3001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "overview",
    "top_k": 10,
    "filters": {
      "source_file": "guide.pdf",
      "chunk_index": 0
    }
  }'
```

Returns only the first chunk from `guide.pdf` (if it matches the query).

**4. Filter by custom metadata:**

```bash
curl -X POST http://localhost:3001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "recent research",
    "top_k": 10,
    "filters": {
      "author": "Alice",
      "year": 2024,
      "category": "AI"
    }
  }'
```

All filters must match (AND logic): author is "Alice" AND year is 2024 AND category is "AI".

**Example Response:**

```json
{
  "query": "machine learning",
  "results": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "content": "Machine learning is a subset of artificial intelligence",
      "score": 0.8234,
      "metadata": {
        "source_file": "textbook.pdf",
        "chunk_index": 3,
        "author": "John Doe"
      }
    }
  ],
  "count": 1
}
```

**Note:** When creating documents with `source_file` and `chunk_index`, both must be provided together. However, when filtering, you can use either one independently or both together.

### **4. Upload a File**

```bash
curl -X POST http://localhost:3001/file/upload \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "filename": "document.pdf",
  "chunks_created": 15,
  "doc_ids": ["uuid1", "uuid2", "...", "uuid15"],
  "status": "indexed"
}
```

### **5. Get Document by ID**

```bash
curl http://localhost:3001/doc/550e8400-e29b-41d4-a716-446655440000
```

### **6. Delete Document**

```bash
curl -X DELETE http://localhost:3001/doc/550e8400-e29b-41d4-a716-446655440000
```

### **7. Save Index**

```bash
curl -X POST http://localhost:3001/index/save
```

**Response:**
```json
{
  "status": "saved",
  "directory": "./data/chroma_data",
  "total_size_mb": 1.45,
  "documents_saved": 100,
  "embeddings_saved": 100,
  "version": "0.9.0"
}
```

### **8. Load Index**

```bash
curl -X POST http://localhost:3001/index/load
```

### **9. Get Metrics**

```bash
curl http://localhost:3001/metrics
```

**Response includes:**
- Index statistics (total documents, total embeddings)
- Performance metrics (query times, percentiles)
- Usage statistics (operations performed)
- Memory consumption
- System information and uptime

### **10. Health Check**

```bash
curl http://localhost:3001/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.9.0"
}
```

### **Python Client Example**

```python
import httpx

BASE_URL = "http://localhost:3001"

# Add document
response = httpx.post(f"{BASE_URL}/doc/add", json={
    "content": "Python is a high-level programming language",
    "metadata": {"topic": "programming"}
})
doc_id = response.json()["id"]

# Search
response = httpx.post(f"{BASE_URL}/search", json={
    "query": "programming languages",
    "top_k": 3
})
results = response.json()["results"]

for result in results:
    print(f"Score: {result['score']:.3f} - {result['content'][:50]}...")
```

---

## Development Process

**TODO: Fill in with dev challenges, learnings, interesting bugs/edge cases discovered, performance optimization insights, etc.**
<!--
Fill in with your experience:
- What challenges did you face during development?
- How did you overcome specific technical hurdles?
- What did you learn about vector databases, embeddings, or FastAPI?
- Any interesting bugs or edge cases you discovered?
- Performance optimization insights?
-->

---

## Known Limitations & Future Improvements

**TODO: Fill in with current limitations, performance improvements, and scalability considerations**
<!--
Fill in with:
- Current limitations you're aware of
- Features you plan to add
- Performance improvements on the roadmap
- Scalability considerations
-->

### Potential Enhancements:
- [ ] Support for more file formats (DOCX, HTML, Markdown)
- [ ] Batch document operations
- [ ] Advanced filtering with operators (>, <, IN, regex)
- [ ] Hybrid search (vector + keyword)
- [ ] Distributed deployment support
- [ ] Custom embedding model configuration
- [ ] Real-time index updates via WebSockets
- [ ] Rate limiting and authentication
- [ ] Multi-tenancy support
- [ ] Vector quantization for memory optimization

---

### Code Style Guidelines
- Follow PEP 8 conventions
- Use type hints for all function signatures
- Write docstrings for all public APIs (Google style)
- Maintain test coverage above 90%
- One logical concept per test case

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **sentence-transformers** team for the excellent embedding models
- **FastAPI** community for the modern web framework
- **PyMuPDF** developers for robust PDF processing
- All contributors to the open-source libraries used in this project

---

## Contact

**Project Repository**: [https://github.com/NickHoulding/vectorforge](https://github.com/NickHoulding/vectorforge)

---

<div align="center">
  <strong>VectorForge - Built using Python and FastAPI</strong>
</div>
