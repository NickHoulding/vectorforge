# VectorForge

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.121.2+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-0.9.0-orange.svg)

> A high-performance in-memory vector database with semantic search capabilities, built with FastAPI and sentence transformers.

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
- Providing a lightweight, embeddable vector database that doesn't require external infrastructure
- Offering real-time indexing and search with comprehensive metrics tracking

---

## What is VectorForge?

VectorForge is a **production-ready, in-memory vector database** designed for semantic search applications. It combines the simplicity of a REST API with the power of transformer-based embeddings to enable intelligent document retrieval.

Unlike traditional databases, VectorForge:
- **Understands meaning** - Uses sentence transformers to encode semantic content
- **Scales efficiently** - In-memory architecture with lazy deletion and automatic compaction
- **Provides observability** - Comprehensive metrics tracking for performance monitoring
- **Supports persistence** - Save and load index state to disk for quick recovery
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
- Real-time embedding generation
- Query performance tracking with percentile metrics (p50, p95, p99)

### **Document Management**
- Add individual documents with custom metadata
- Retrieve documents by ID
- Delete documents with lazy deletion strategy
- Automatic embedding generation and normalization
- Content length validation (max 10,000 characters)

### **File Processing**
- **PDF Support** - Text extraction from PDF documents
- **TXT Support** - UTF-8 text file processing
- **Automatic Chunking** - Configurable chunk size (default: 500 chars) with overlap (default: 50 chars)
- **Batch Processing** - Upload files and get multiple document chunks
- **File-level Deletion** - Remove all chunks from a specific file

### **Index Management**
- **Lazy Deletion** - Mark documents as deleted without immediate removal
- **Automatic Compaction** - Triggered when deleted ratio exceeds threshold (default: 25%)
- **Manual Rebuild** - Reconstruct entire index from active documents
- **Persistence** - Save/load index to disk with metadata and embeddings
- **Index Statistics** - Real-time metrics on document counts and health

### **Comprehensive Metrics**
- **Performance Metrics** - Query times, averages, min/max, percentiles
- **Usage Metrics** - Documents added/deleted, files uploaded, chunks created
- **Memory Metrics** - Embeddings size, documents size, total memory usage
- **System Info** - Model name, embedding dimension, uptime, version
- **Timestamps** - Track last query, document addition, compaction, file upload

### **Production-Ready**
- RESTful API with OpenAPI documentation
- Comprehensive error handling with detailed HTTP status codes
- Health check endpoint for monitoring
- Version tracking across all operations
- Extensive test coverage (~100% for core endpoints)

---

## Architecture

### **Design Patterns & Technical Decisions**

#### **1. In-Memory Vector Store**
VectorForge uses an in-memory architecture for maximum performance:
- **Documents Storage** - `dict[str, dict]` mapping doc IDs to content/metadata
- **Embeddings Storage** - `list[np.ndarray]` of normalized 384-dimensional vectors
- **Index Mappings** - Bidirectional mappings between positions and document IDs
- **Deleted Documents** - `set[str]` for lazy deletion tracking

**Rationale**: Eliminates I/O overhead for sub-millisecond search latency.

#### **2. Lazy Deletion Strategy**
Documents are marked as deleted rather than immediately removed:
```python
deleted_docs.add(doc_id)  # Mark only
# Actual removal happens during compaction
```

**Benefits**:
- O(1) deletion operations
- Batch cleanup reduces index fragmentation
- Configurable compaction threshold (default: 25% deleted ratio)

#### **3. Automatic Compaction**
The engine automatically triggers compaction when deletion threshold is exceeded:
```python
if len(deleted_docs) / len(embeddings) > 0.25:
    compact()  # Rebuild index, remove deleted docs
```

**Rationale**: Balances performance with memory efficiency.

#### **4. Normalized Embeddings**
All embeddings are L2-normalized during creation:
```python
embedding = embedding / np.linalg.norm(embedding)
```

**Benefits**:
- Cosine similarity reduces to dot product during search operations (faster)
- Numerical stability in similarity calculations

#### **5. Modular Architecture**
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

#### **6. Comprehensive Metrics Tracking**
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

#### **7. Metadata Coupling Validation**
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

### **3. Upload a File**

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

### **4. Get Document by ID**

```bash
curl http://localhost:3001/doc/550e8400-e29b-41d4-a716-446655440000
```

### **5. Delete Document**

```bash
curl -X DELETE http://localhost:3001/doc/550e8400-e29b-41d4-a716-446655440000
```

### **6. Save Index to Disk**

```bash
curl -X POST http://localhost:3001/index/save
```

**Response:**
```json
{
  "status": "saved",
  "directory": "./data",
  "metadata_size_mb": 0.25,
  "embeddings_size_mb": 1.2,
  "total_size_mb": 1.45,
  "documents_saved": 100,
  "embeddings_saved": 100
}
```

### **7. Load Index from Disk**

```bash
curl -X POST http://localhost:3001/index/load
```

### **8. Get Metrics**

```bash
curl http://localhost:3001/metrics
```

**Response includes:**
- Index statistics (documents, embeddings, deleted ratio)
- Performance metrics (query times, percentiles)
- Usage statistics (operations performed)
- Memory consumption
- System information and uptime

### **9. Health Check**

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
- [ ] Advanced filtering and metadata-based search
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
