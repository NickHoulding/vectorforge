# VectorForge

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.3-orange.svg)

> A vector database microservice with semantic search capabilities, powered by ChromaDB, built with FastAPI and sentence transformers.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [What is VectorForge?](#what-is-vectorforge)
- [Interactive Demo](#interactive-demo)
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [HNSW Configuration](#hnsw-configuration)
- [Development Process](#development-process)
- [Known Limitations](#known-limitations)
- [Related Projects](#related-projects)
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

VectorForge is a vector database designed for semantic search applications. It combines the simplicity of a REST API with the power of transformer-based embeddings and ChromaDB's persistent storage to enable intelligent document retrieval.

Unlike traditional databases, VectorForge:
- **Understands meaning** - Uses sentence transformers to encode semantic content
- **Provides observability** - Comprehensive metrics tracking for performance monitoring
- **Handles documents smartly** - Automatic chunking, metadata preservation, and file processing

Perfect for building:
- Semantic search engines
- Document Q&A systems
- Knowledge base retrieval
- Content recommendation systems
- RAG (Retrieval-Augmented Generation) backends

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
- **langchain-text-splitters** - `RecursiveCharacterTextSplitter` for document chunking
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

---

## Key Features

### **Semantic Search**
- Vector similarity search using cosine distance
- Configurable top-k results
- **Metadata filtering** - Filter results by exact equality or operator expressions (`$gte`, `$lte`, `$ne`, `$in`) with AND logic; filter on document text with `document_filter` (`$contains`, `$not_contains`)
- Real-time embedding generation
- Query performance tracking with percentile metrics (p50, p95, p99)

### **Document Management**
- Add individual documents with custom metadata
- Add multiple documents in a single batch request (up to 100 per call)
- Retrieve documents by ID
- Delete one or multiple documents in a single batch request
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

#### **3. Pre-Normalized Embeddings**
All embeddings are L2-normalized using numpy during creation:
```python
embedding = embedding / np.linalg.norm(embedding)
```

**Benefits**:
- Cosine similarity reduces to dot product during search operations (faster)
- Numerical stability in similarity calculations

#### **4. Modular Architecture**
```
vectorforge/
├── api/                # FastAPI routers (presentation layer)
│   ├── collections.py  # Collection CRUD
│   ├── documents.py    # Document CRUD
│   ├── files.py        # File upload
│   ├── search.py       # Semantic search
│   ├── index.py        # Stats & HNSW config
│   ├── system.py       # Health & metrics
│   ├── config.py       # Collection metadata config
│   └── decorators.py   # Error handling & auth decorators
├── models/             # Pydantic models (data layer)
│   ├── collections.py
│   ├── documents.py
│   ├── files.py
│   ├── search.py
│   ├── index.py
│   ├── metrics.py
│   └── metadata.py
├── vector_engine.py    # Core vector operations (business logic)
├── collection_manager.py
├── doc_processor.py    # Text extraction and chunking
└── metrics_store.py    # SQLite-backed metrics persistence
```

**Rationale**: Separation of concerns enables testing and maintainability.

#### **5. Comprehensive Metrics Tracking**
VectorForge uses a dataclass-based metrics system:
```python
@dataclass
class EngineMetrics:
    total_queries: int
    query_times: deque[float]
    # Other tracked metrics...
```

**Benefits**:
- Zero-overhead tracking (in-memory counters)
- Historical query performance analysis
- Production debugging and optimization

#### **6. Metadata Coupling Validation**
File-based documents require both `source` and `chunk_index`:
```python
if has_source != has_chunk_index:
    raise ValueError("Must contain both or neither")
```

**Rationale**: Maintains data integrity for file reconstruction.

---

## Interactive Demo

The `demo/` directory contains an interactive REPL that lets you exercise every API endpoint against a live VectorForge instance directly from the terminal. No curl commands or external tooling required.

### **How it works**

1. On startup the demo checks whether the VectorForge API is accessible. If it isn't, it runs `docker compose up -d` automatically to start the API container and waits for it to become live before proceeding.
2. A menu of 19 feature keys is displayed. Type any key to invoke that endpoint. The demo will prompt you for the required parameters, fire the request, and pretty-print the JSON response.
3. On exit you are asked what to do with the API container: leave it running, stop it, remove it, or remove it along with all volume data.

### **Prerequisites**

- Docker (with the Compose plugin)
- Python 3.11+ with `uv` and dependencies installed (see [Getting Started](#getting-started))

### **Running the demo**

```bash
# From the project root:
uv run demo
```

### **Feature menu**

```
collections:create          Create a new collection
collections:list            List all collections
collections:get             Get details for one collection
collections:delete          Delete a collection

documents:add               Add a single document
documents:get               Fetch a document by ID
documents:batch_add         Add multiple documents
documents:delete            Delete a document by ID
documents:batch_delete      Batch-delete documents by ID list

files:upload                Upload and index a .pdf or .txt file
files:list                  List indexed files in a collection
files:delete                Delete all chunks for a file

search                      Semantic similarity search

index:stats                 Get index statistics and HNSW config
index:update_hnsw           Migrate HNSW index configuration

system:health               Basic health check
system:health_ready         Readiness probe
system:health_live          Liveness probe
system:metrics              Comprehensive collection metrics
```

Type `help` at any time to reprint the menu, and `quit` to exit.

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
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

3. **Verify installation**
```bash
# Should print current version of vectorforge:
uv run python -c "import vectorforge; print(vectorforge.__version__)"
```

### **How to Run**

#### **Start the API Server**
```bash
uv run vectorforge-api
```

The API will be available at:
- **Base URL**: http://localhost:3001
- **Interactive Docs**: http://localhost:3001/docs
- **OpenAPI Schema**: http://localhost:3001/openapi.json

#### **Run Tests**
```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=vectorforge --cov-report=html

# Run specific test file
uv run pytest tests/test_doc_endpoints.py -v

# Run tests matching pattern
uv run pytest tests/ -k "search" -v
```

#### **View Coverage Report**
```bash
uv run pytest tests/ --cov=vectorforge --cov-report=html
xdg-open htmlcov/index.html  # Linux
open htmlcov/index.html      # macOS
```

### **Environment Variables**

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `3001` | Port for FastAPI server |
| `API_HOST` | `0.0.0.0` | Network interface the API binds to |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `CHROMA_DATA_DIR` | `./data/chroma` | Directory for ChromaDB persistent storage |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace model cache directory |
| `MAX_COLLECTIONS` | `100` | Maximum number of collections allowed |
| `COLLECTION_CACHE_SIZE` | `50` | Number of collection engines held in the FIFO cache |

---

## Usage

### **1. Add a Document**

```python
import requests

response = requests.post(
    "http://localhost:3001/collections/vectorforge/documents",
    json={
        "content": "Machine learning is a subset of artificial intelligence",
        "metadata": {
            "source": "ml_intro.txt",
            "chunk_index": 0,
            "author": "John Doe",
        },
    },
)
doc_id = response.json()["id"]
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "indexed"
}
```

### **2. Search Documents**

```python
response = requests.post(
    "http://localhost:3001/collections/vectorforge/search",
    json={"query": "What is AI?", "top_k": 5},
)
results = response.json()["results"]
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
        "source": "ml_intro.txt",
        "chunk_index": 0,
        "author": "John Doe"
      }
    }
  ],
  "count": 1
}
```

### **3. Search with Metadata Filters**

Filter search results using exact equality matching, operator expressions, or document-text
filtering. All `filters` conditions use AND logic (all must match).

**VectorForge uses ChromaDB under the hood.** The `filters` parameter (API) and `where` parameter
(MCP) are passed directly to ChromaDB's `where` clause. VectorForge supports the following
ChromaDB operator expressions: `$gte`, `$lte`, `$ne`, `$in` for metadata filtering, and
`$contains`, `$not_contains` for document text filtering. Refer to ChromaDB documentation to
learn more about where clause syntax.

**Note:** When using the VectorForge MCP Server, the `filters` parameter is exposed as
`where` in the `search_documents` tool for semantic consistency with ChromaDB terminology.
Both accept the same dict structure shown in the examples below.

**Key Features:**
- **Exact matching**: Filters use case-sensitive equality comparison
- **Operator expressions**: `$gte`, `$lte`, `$ne`, `$in` are supported inside `filters`
- **Document-text filtering**: `document_filter` with `$contains` or `$not_contains` filters on document content
- **AND logic**: Multiple filters require all to match
- **Flexible filtering**: Filter by any metadata field independently
- **No pairing required**: `source` and `chunk_index` can be used separately or together
- **Performance**: Filtering happens after similarity scoring

#### **Common Filtering Scenarios:**

**1. Filter by source file (all chunks from a specific file):**

```python
response = requests.post(
    "http://localhost:3001/collections/vectorforge/search",
    json={
        "query": "machine learning concepts",
        "top_k": 10,
        "filters": {"source": "textbook.pdf"},
    },
)
```

Returns all matching chunks from `textbook.pdf`, regardless of chunk index.

**2. Filter by chunk index (first chunks from all files):**

```python
response = requests.post(
    "http://localhost:3001/collections/vectorforge/search",
    json={
        "query": "introduction",
        "top_k": 10,
        "filters": {"chunk_index": 0},
    },
)
```

Returns all matching first chunks (index 0) from any file. Useful for finding document introductions.

**3. Filter by both (specific chunk from specific file):**

```python
response = requests.post(
    "http://localhost:3001/collections/vectorforge/search",
    json={
        "query": "overview",
        "top_k": 10,
        "filters": {"source": "guide.pdf", "chunk_index": 0},
    },
)
```

Returns only the first chunk from `guide.pdf` (if it matches the query).

**4. Filter by custom metadata:**

```python
response = requests.post(
    "http://localhost:3001/collections/vectorforge/search",
    json={
        "query": "recent research",
        "top_k": 10,
        "filters": {"author": "Alice", "year": 2024, "category": "AI"},
    },
)
```

All filters must match (AND logic): author is "Alice" AND year is 2024 AND category is "AI".

**5. Range and set operators in `filters`:**

```python
response = requests.post(
    "http://localhost:3001/collections/vectorforge/search",
    json={
        "query": "deep learning",
        "top_k": 10,
        "filters": {
            "year": {"$gte": 2022},
            "category": {"$in": ["AI", "ML"]},
            "status": {"$ne": "draft"},
        },
    },
)
```

Supported metadata operators: `$gte`, `$lte`, `$ne`, `$in`. The `$in` value must be a list of
`str`, `int`, `float`, or `bool` values. Other operator expressions return HTTP 422.

**6. Document-text filtering with `document_filter`:**

```python
response = requests.post(
    "http://localhost:3001/collections/vectorforge/search",
    json={
        "query": "neural networks",
        "top_k": 10,
        "document_filter": {"$contains": "Python"},
    },
)
```

`document_filter` filters on the document text content (not metadata). Supported operators:
`$contains` and `$not_contains`. Can be combined with `filters`.

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
        "source": "textbook.pdf",
        "chunk_index": 3,
        "author": "John Doe"
      }
    }
  ],
  "count": 1
}
```

**Note:** When creating documents with `source` and `chunk_index`, both must be provided together. However, when filtering, they can be used either independently or together.

### **4. Upload a File**

```python
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:3001/collections/vectorforge/files/upload",
        files={"file": f},
    )
result = response.json()
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

```python
doc_id = "550e8400-e29b-41d4-a716-446655440000"
response = requests.get(
    f"http://localhost:3001/collections/vectorforge/documents/{doc_id}"
)
document = response.json()
```

### **6. Delete Document**

```python
doc_id = "550e8400-e29b-41d4-a716-446655440000"
response = requests.delete(
    f"http://localhost:3001/collections/vectorforge/documents/{doc_id}"
)
```

### **6a. Batch Add Documents**

```python
response = requests.post(
    "http://localhost:3001/collections/vectorforge/documents/batch",
    json={
        "documents": [
            {
                "content": "Machine learning is a subset of artificial intelligence",
                "metadata": {"source": "ml_intro.txt", "chunk_index": 0},
            },
            {
                "content": "Deep learning uses multi-layered neural networks",
                "metadata": {"source": "ml_intro.txt", "chunk_index": 1},
            },
        ]
    },
)
doc_ids = response.json()["ids"]
```

**Response:**
```json
{
  "ids": [
    "550e8400-e29b-41d4-a716-446655440000",
    "661f9511-f30c-52e5-b827-557766551111"
  ],
  "status": "indexed"
}
```

Batches are capped at `MAX_BATCH_SIZE` (default 100). Requests exceeding this limit return HTTP 422.

### **6b. Batch Delete Documents**

```python
response = requests.delete(
    "http://localhost:3001/collections/vectorforge/documents",
    json={
        "ids": [
            "550e8400-e29b-41d4-a716-446655440000",
            "661f9511-f30c-52e5-b827-557766551111",
        ]
    },
)
deleted_ids = response.json()["ids"]
```

**Response:**
```json
{
  "ids": [
    "550e8400-e29b-41d4-a716-446655440000",
    "661f9511-f30c-52e5-b827-557766551111"
  ],
  "status": "deleted"
}
```

Only IDs that actually existed are returned. If none of the requested IDs exist, HTTP 404 is returned. Batches are capped at `MAX_BATCH_SIZE` (default 100).

### **7. Get Metrics**

```python
response = requests.get("http://localhost:3001/collections/vectorforge/metrics")
metrics = response.json()
```

**Response includes:**
- Index statistics (total documents, total embeddings)
- Performance metrics (query times, percentiles)
- Usage statistics (operations performed)
- Memory consumption
- System information and uptime

### **8. Health Check**

```python
response = requests.get("http://localhost:3001/health")
print(response.json())
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.3",
  "chromadb_heartbeat": 1234567890,
  "total_collections": 1
}
```

---

## HNSW Configuration

VectorForge uses ChromaDB's **Hierarchical Navigable Small World (HNSW)** algorithm for efficient approximate nearest neighbor search. You can tune HNSW parameters to optimize the tradeoff between search accuracy and performance.

### **Getting Current Configuration**

```python
response = requests.get("http://localhost:3001/collections/vectorforge/stats")
stats = response.json()
```

**Response includes:**
```json
{
  "status": "success",
  "total_documents": 1250,
  "embedding_dimension": 384,
  "hnsw_config": {
    "space": "cosine",
    "ef_construction": 100,
    "ef_search": 100,
    "max_neighbors": 16,
    "resize_factor": 1.2,
    "sync_threshold": 1000
  }
}
```

### **Updating HNSW Configuration**

Update HNSW parameters via zero-downtime collection migration:

```python
response = requests.put(
    "http://localhost:3001/collections/vectorforge/config/hnsw",
    params={"confirm": "true"},
    json={"ef_search": 150, "max_neighbors": 32},
)
result = response.json()
```

**Response:**
```json
{
  "status": "success",
  "message": "HNSW configuration updated successfully",
  "migration": {
    "documents_migrated": 1250,
    "time_taken_seconds": 8.47,
    "old_collection_deleted": true,
    "temp_verified": true
  },
  "config": {
    "space": "cosine",
    "ef_construction": 100,
    "ef_search": 150,
    "max_neighbors": 32,
    "resize_factor": 1.2,
    "sync_threshold": 1000
  }
}
```

### **How HNSW Migration Works**

The migration performs a **non-destructive collection-level blue-green style migration**:

1. Creates a temporary collection with the updated HNSW settings
2. Migrates all documents + embeddings from the original to the temp collection in batches (1000 docs/batch)
3. Verifies the temp collection document count matches the original. **The original is preserved until this check passes.**
4. Deletes the original collection (safe: temp is verified)
5. Creates the final collection under the original name
6. Migrates documents from temp to final, then verifies the final count
7. Deletes the temp collection and swaps the engine reference

**Note:** This is collection-level migration within the same ChromaDB database. All collections share the same Docker volume (`vectorforge-data:/data`).

### **Important Considerations**

- **Disk Space**: Migration temporarily requires up to **3x disk space** (old + temp + new collections)

- **Shared Storage**: All collections use the same persistent volume. This is **not infrastructure-level** blue-green deployment with separate volumes.

- **Single Container**: If running multiple replicas, scale to 1 container before migrating to avoid race conditions.

- **Confirmation Required**: Must include `?confirm=true` query parameter as a safety gate.

### **HNSW Parameters Explained**

| Parameter | Description | Higher Value = | Typical Range |
|-----------|-------------|----------------|---------------|
| `space` | Distance metric | N/A (cosine, l2, ip) | - |
| `ef_construction` | Build-time search depth | Better index quality, slower construction | 100-500 |
| `ef_search` | Query-time search depth | Higher accuracy, slower queries | 10-500 |
| `max_neighbors` | Connections per node (M) | Better recall, more memory | 16-64 |
| `resize_factor` | Index growth multiplier | Less frequent resizing | 1.2-2.0 |
| `sync_threshold` | Batch size for persistence | Less frequent disk writes | 100-10000 |

### **When to Tune HNSW**

**Good for:**
- Initial performance tuning after deployment
- Adjusting search quality vs speed tradeoff
- Experimenting with distance metrics

**Not recommended for:**
- Frequent configuration changes (expensive to recreate)
- Very large datasets (>10M docs) without disk space planning
- Multi-replica deployments (coordinate manually first)

---

## Development Process

VectorForge went through a significant architectural evolution during development:

### **Initial Approach: C++/Pybind11 Optimization**
The project originally explored a C++ implementation with Python bindings via pybind11 for maximum performance. This approach included:
- Custom C++ vector operations and similarity calculations
- CMake build system for cross-platform compilation
- Python bindings for seamless integration

### **Pivot to ChromaDB**
After evaluating the tradeoffs, the project pivoted to ChromaDB as the core vector database engine. This decision was driven by the follwing benefits:

- **Production-readiness** - Battle-tested persistence and ACID guarantees
- **Built-in optimizations** - HNSW indexing, efficient similarity search
- **Maintained by experts** - Active development by vector DB specialists
- **Less complexity** - No C++ compilation, easier deployment
- **Rich features** - Metadata filtering, collections, batch operations
- **Docker-friendly** - Simpler containerization without build toolchains

**Tradeoffs:**
- Slightly higher memory overhead vs. custom C++
- Less control over low-level optimizations

**Outcome:** The ChromaDB implementation achieved production-ready stability while maintaining excellent performance for the target use cases. The code from the C++ exploration is preserved in git history (commits before the ChromaDB migration) for reference.

### **Key Technical Decisions**
1. **FastAPI over Flask** - Async support for high concurrency
2. **ChromaDB for persistence** - Eliminates custom storage implementation
3. **Sentence Transformers** - Industry-standard embeddings
4. **Comprehensive testing** - 500+ test pytest suite covering all functionality
5. **Docker-first deployment** - Built for containerized environments

### **Development Insights**
- Vector databases are complex - using a specialized solution (ChromaDB) proved more reliable than building from scratch
- Type hints and Pydantic validation catch errors early and improve API usability
- Comprehensive metrics tracking is essential for production debugging
- Docker multi-stage builds dramatically reduce image size while maintaining functionality

---

## Known Limitations

#### Metadata
- **No `None` values in metadata.** ChromaDB rejects `None` as a metadata value. VectorForge
  validates metadata before insertion and returns HTTP 422 with a descriptive error when `None`
  is detected, rather than forwarding the value to ChromaDB.
- **List metadata values cannot be filtered.** ChromaDB stores list values but does not support
  membership testing in `where` clauses. A filter like `{"tags": "python"}` returns zero results
  even if the document has `{"tags": ["python", "ml"]}`. Tags and similar multi-value fields must
  be stored as space-joined strings and filtered accordingly.
- **No nested metadata objects.** ChromaDB only accepts `str`, `int`, `float`, and `bool` as leaf
  metadata values. Nested dicts are not supported.

#### Search and Filtering
- **AND-only filter logic.** The `filters` parameter always combines multiple conditions with
  `$and`. There is no way to express OR conditions through the current API, even though ChromaDB
  supports them natively.
- **No `$contains` on metadata fields.** ChromaDB's `$contains` operator only works against
  document text, not metadata. Use `document_filter: {"$contains": "..."}` for text-content
  filtering and `filters` with operator expressions (`$gte`, `$lte`, `$ne`, `$in`) for metadata.
- **`top_k` is capped at 100.** `MAX_TOP_K = 100` is enforced at the API layer. Requests
  requiring more than 100 results must be paginated manually.

#### File Processing
- **PDF and TXT only.** No support for `.docx`, `.md`, `.html`, `.csv`, `.rtf`, or other formats.
- **No OCR.** PDF extraction reads the text layer only. Scanned PDFs (image-only) return no
  content and produce a 400 error.
- **UTF-8 only for text files.** `.txt` files with Latin-1 or other encodings produce an HTTP 500
  instead of a helpful error message.
- **No per-chunk size limit in file uploads.** The `MAX_CONTENT_LENGTH` (10,000 chars) is
  enforced by the Pydantic model for `POST /collections/{name}/documents` but not for the file upload path. The upload
  endpoint accepts `chunk_size` and `chunk_overlap` parameters (defaulting to 500 and 50 chars),
  but does not validate that individual chunks stay within `MAX_CONTENT_LENGTH`.

#### Performance
- **Synchronous SQLite write on every operation.** Every `add_docs`, `delete_docs`, and `search`
  call triggers a blocking SQLite round-trip (open connection → WAL pragma → UPDATE → commit →
  close). All write paths: batch add, batch delete, file upload, and file delete consolidate to
  a single `save()` call per request regardless of how many documents are involved. There is no
  async write path.
- **`GET /metrics` scans the full data directory across all collections.** `_get_chromadb_disk_size()` calls
  `os.walk()` across the entire ChromaDB data directory. In a multi-collection deployment this
  scans data for all collections, not just the requested one. Results are cached for
  `DISK_SIZE_TTL_MINS` (default 5 minutes) to limit scan frequency.
- **No concurrent request safety for in-memory counters.** The engine's metrics counters
  (`total_queries`, `docs_added`, etc.) are plain integers with no locking. Concurrent requests
  share a single `VectorEngine` instance and can produce lost updates on these counters.

#### HNSW Migration
- **Blocks the calling request.** `PUT /index/config/hnsw` runs the migration synchronously in
  the request handler. Large indexes may time out at the HTTP layer before the migration
  completes.
- **Requires up to 3× disk space.** At peak, three copies of the index exist simultaneously
  (original, temp, and final collections).
- **Partial rollback window after original deletion.** The original collection is preserved until
  the temp collection is fully populated and its document count verified. Failures during temp
  population leave the original intact. However, if a failure occurs after the original is deleted
  (between the delete and the final collection being verified), that data cannot be automatically
  restored. Recovery in that window requires a backup or a clean re-index.

#### Configuration
- **`MODEL_NAME` is hardcoded.** The embedding model (`all-MiniLM-L6-v2`) cannot be changed via
  environment variable. Switching models requires a code change and a full re-index because stored
  embeddings are dimension-specific (384d).
- **Chunk size is API-configurable via** `chunk_size` **and** `chunk_overlap` **parameters on the file upload endpoint.** `DEFAULT_CHUNK_SIZE` (500) and `DEFAULT_CHUNK_OVERLAP` (50) are used when omitted.

#### Deployment
- **Single-process only.** The `migration_in_progress` flag and in-memory metrics state are
  per-process. Running multiple uvicorn workers or replicas will cause split state. Scale to
  one replica before running HNSW migrations.
- **Docker health check does not verify readiness.** The Dockerfile's `HEALTHCHECK` calls
  `/health/live`, which returns `{"status": "alive"}` unconditionally without verifying that
  ChromaDB is accessible or the model is loaded. Use `/health/ready` for a meaningful readiness
  check.

---

## Related Projects

- **VectorForge MCP Server** - MCP adapter for AI assistant integration ([README](vectorforge_mcp/README.md))

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <strong>VectorForge</strong>
</div>
