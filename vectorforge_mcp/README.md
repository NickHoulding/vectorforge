# VectorForge MCP Server

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastMCP](https://img.shields.io/badge/FastMCP-2.14+-green.svg)
![Version](https://img.shields.io/badge/version-1.0.3-orange.svg)

> Model Context Protocol server for VectorForge - Enable semantic search capabilities in AI assistants like Claude Desktop.

---

## Table of Contents

- [What is VectorForge MCP Server?](#what-is-vectorforge-mcp-server)
- [Why MCP?](#why-mcp)
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Available Tools](#available-tools)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## What is VectorForge MCP Server?

The VectorForge MCP Server is a **Model Context Protocol (MCP) adapter** that exposes VectorForge's semantic search capabilities to AI assistants and other MCP-compatible clients. It acts as a bridge between the VectorForge REST API and MCP clients, enabling AI assistants to:

- Create and manage collections for multi-tenancy or domain separation
- Index and search documents semantically
- Upload and process files (PDF, TXT)
- Retrieve comprehensive system metrics

**Key Components:**
- **MCP Server** - Implements the Model Context Protocol specification
- **Tool Adapters** - Wraps VectorForge API endpoints as MCP tools
- **Error Handling** - Provides consistent error responses with detailed diagnostics
- **Logging** - Configurable logging for monitoring and debugging

---

## Why MCP?

The Model Context Protocol (MCP) is an open standard that enables AI assistants to interact with external tools and data sources. By implementing an MCP server for VectorForge, we enable:

### **Benefits**

1. **AI Assistant Integration** - Use VectorForge directly from Claude Desktop, IDEs, and other MCP clients
2. **Natural Language Interface** - AI assistants can perform semantic search through natural conversation
3. **Tool Discoverability** - MCP clients automatically discover and understand available tools
4. **Standardized Protocol** - No custom integrations needed for each AI platform
5. **Separation of Concerns** - MCP server is independent of core VectorForge API

### **Use Cases**

- **Document Q&A** - AI assistants can search your indexed documents to answer questions
- **Knowledge Base** - Build a searchable knowledge base accessible to AI
- **Research Assistant** - Index research papers and query them semantically
- **Code Search** - Index code documentation and search by concepts
- **Content Discovery** - Find relevant content based on semantic similarity

---

## Tech Stack

### **Core Technologies**
- **Python 3.11+** - Modern Python with type hints throughout
- **FastMCP 2.14+** - Fast, modern MCP server framework
- **VectorForge API** - Core vector database REST API (required, runs separately)

### **Dependencies**
- **requests** - HTTP client for VectorForge API communication

---

## Key Features

### **16 MCP Tools**
Organized into 6 categories:

**Collections** (4 tools)
- List all collections
- Get collection details
- Create a collection with optional HNSW configuration
- Delete a collection

**Documents** (5 tools)
- Get document by ID
- Add document with metadata
- Batch add documents
- Delete document
- Batch delete documents

**Files** (3 tools)
- List indexed files
- Upload and process files
- Delete all chunks from a file

**Index** (1 tool)
- Get index statistics and HNSW configuration

**Search** (1 tool)
- Semantic search with top-k results and metadata filtering

**System** (2 tools)
- Get comprehensive metrics
- Health check

All tools (except `check_health`) accept a `collection_name` parameter, defaulting to `"vectorforge"`.

### **Production-Ready Features**
- **Comprehensive Error Handling** - Catches network errors, timeouts, and HTTP error responses
- **Logging** - Configurable logging with timestamps and levels
- **Configuration** - Centralized config with validation
- **Type Safety** - Full type hints throughout codebase

---

## Architecture

### **Design Overview**

```
┌─────────────────┐
│  MCP Client     │ (Claude Desktop, IDE, etc.)
│  (AI Assistant) │
└────────┬────────┘
         │ MCP Protocol
         │
┌────────▼────────┐
│  VectorForge    │
│  MCP Server     │ ← You are here
├─────────────────┤
│ • 16 MCP Tools  │
│ • Error Handler │
│ • Logging       │
└────────┬────────┘
         │ HTTP REST
         │
┌────────▼────────┐
│  VectorForge    │
│  API            │ (Docker container, port 3001)
├─────────────────┤
│ • ChromaDB      │
│ • Embeddings    │
│ • Storage       │
└─────────────────┘
```

### **Module Structure**

```
vectorforge_mcp/
├── __init__.py
├── __main__.py        # Entry point and main()
├── client.py          # HTTP client helpers (get, post, put, delete)
├── config.py          # Configuration settings
├── decorators.py      # Error handling decorator
├── instance.py        # FastMCP server instance
├── utils.py           # Response builders
└── tools/             # MCP tool implementations
    ├── collections.py # Collection management tools
    ├── documents.py   # Document management tools
    ├── files.py       # File upload/management tools
    ├── index.py       # Index statistics tool
    ├── search.py      # Semantic search tool
    └── system.py      # System info tools
```

### **Error Handling Strategy**

All tools are wrapped with `@handle_tool_errors` decorator that catches:
- **Connection Errors** - API not running, network issues
- **Timeouts** - Request timeouts with helpful messages
- **HTTP Errors** - Non-2xx responses surfaced with status code and body
- **Generic Exceptions** - Fallback error handling

This provides consistent error responses across all tools without duplicate code.

---

## Getting Started

### **Prerequisites**

1. **VectorForge API must be running**
   ```bash
   # From the project root:
   docker compose up -d
   ```

2. **Python 3.11+** installed
3. **uv** package manager

### **Installation**

```bash
# Navigate to project root
cd vectorforge

# Install with MCP dependencies
uv sync --group mcp
```

### **Verify Installation**

```bash
# Check if command is available
which vectorforge-mcp

# Test configuration validation
python -c "from vectorforge_mcp.config import MCPConfig; MCPConfig.validate(); print('Config valid')"
```

### **Running the Server**

```bash
# Start both the API and MCP server together
docker compose up -d
```

The MCP SSE endpoint will be available at `http://localhost:3002/sse`.

For local development without Docker:

```bash
uv run vectorforge-mcp
```

---

## Usage

### **Connecting from Claude Desktop**

1. **Configure Claude Desktop**

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or equivalent:

```json
{
  "mcpServers": {
    "vectorforge": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "http://localhost:3002/sse"]
    }
  }
}
```

With `docker compose up -d` running, `mcp-remote` bridges Claude Desktop (which requires a `command`-based stdio entry) to the containerized SSE server — no local Python or `uv` installation required. Node.js must be installed on the host for `npx` to work.

2. **Restart Claude Desktop**

3. **Verify Connection**

In Claude, ask: "What VectorForge tools are available?"

Claude should list all 16 MCP tools.

### **Connecting from Other MCP Clients**

The server uses SSE transport. Point your MCP client at `http://localhost:3002/sse`. Refer to your client's documentation for connection setup.

---

## Available Tools

### **Collections**

#### `list_collections`
Get all collections with their names, IDs, document counts, and metadata.

**Example:**
```
"List all VectorForge collections"
```

#### `get_collection`
Get detailed information about a specific collection including document count and HNSW config.

**Parameters:**
- `collection_name` (str) - Name of the collection

**Example:**
```
"Get details for the 'vectorforge' collection"
```

#### `create_collection`
Create a new collection for multi-tenancy or domain separation. Optionally configure HNSW parameters.

**Parameters:**
- `collection_name` (str) - Collection name (alphanumeric, underscores, hyphens)
- `description` (str, optional) - Collection description
- `hnsw_space` (str, optional) - Distance metric: `"cosine"`, `"l2"`, or `"ip"` (default: `"cosine"`)
- `hnsw_ef_construction` (int, optional) - Build-time search depth (default: 100)
- `hnsw_ef_search` (int, optional) - Query-time search depth (default: 100)
- `hnsw_max_neighbors` (int, optional) - Connections per node (default: 16)
- `hnsw_resize_factor` (float, optional) - Index growth multiplier (default: 1.2)
- `hnsw_sync_threshold` (int, optional) - Batch size for persistence (default: 1000)
- `metadata` (dict, optional) - Custom metadata (max 20 key-value pairs)

**Example:**
```
"Create a collection named 'research' with description 'Research papers'"
```

#### `delete_collection`
Permanently delete a collection and all its documents. Cannot delete the default `vectorforge` collection.

**Parameters:**
- `collection_name` (str) - Name of the collection to delete
- `confirm` (bool) - Must be `true` to confirm deletion

**Example:**
```
"Delete the 'research' collection"
```

---

### **Documents**

#### `get_document`
Fetch document content and metadata by ID.

**Parameters:**
- `doc_id` (str) - Unique document identifier (UUID)
- `collection_name` (str, optional) - Collection name (default: `"vectorforge"`)

**Example:**
```
"Get document 550e8400-e29b-41d4-a716-446655440000"
```

#### `add_document`
Index text content for semantic search. Generates embeddings automatically.

**Parameters:**
- `content` (str) - Document text content (required, non-empty)
- `metadata` (dict, optional) - Custom metadata
- `collection_name` (str, optional) - Collection name (default: `"vectorforge"`)

**Example:**
```
"Add a document with content 'Python is a programming language' and metadata {\"topic\": \"tech\"}"
```

#### `delete_document`
Permanently remove a document and its embeddings.

**Parameters:**
- `doc_id` (str) - Document ID to delete
- `collection_name` (str, optional) - Collection name (default: `"vectorforge"`)

**Example:**
```
"Delete document 550e8400-e29b-41d4-a716-446655440000"
```

#### `batch_add_documents`
Index multiple documents in a single request. More efficient than adding one at a time — all documents are embedded and persisted atomically.

**Parameters:**
- `documents` (list) - List of document objects, each with a `content` key (str) and optional `metadata` key (dict)
- `collection_name` (str, optional) - Collection name (default: `"vectorforge"`)

**Example:**
```
"Add these three documents to the index: ..."
```

#### `batch_delete_documents`
Permanently remove multiple documents and their embeddings in a single request. IDs that do not exist are silently ignored.

**Parameters:**
- `doc_ids` (list) - List of document UUIDs to delete
- `collection_name` (str, optional) - Collection name (default: `"vectorforge"`)

**Example:**
```
"Delete documents abc-123, def-456, and ghi-789"
```

---

### **Files**

#### `list_files`
Get all filenames that have been uploaded and chunked.

**Parameters:**
- `collection_name` (str, optional) - Collection name (default: `"vectorforge"`)

**Example:**
```
"List all indexed files"
```

#### `upload_file`
Upload and index a file (PDF, TXT). Extracts text, chunks it, generates embeddings.

**Parameters:**
- `file_path` (str) - Absolute path to file
- `collection_name` (str, optional) - Collection name (default: `"vectorforge"`)
- `chunk_size` (int, optional) - Maximum characters per chunk (default: 500)
- `chunk_overlap` (int, optional) - Overlapping characters between chunks (default: 50)

**Example:**
```
"Upload file /path/to/document.pdf"
```

#### `delete_file`
Delete all document chunks from a specific uploaded file.

**Parameters:**
- `filename` (str) - Name of the source file
- `collection_name` (str, optional) - Collection name (default: `"vectorforge"`)

**Example:**
```
"Delete file document.pdf"
```

---

### **Index**

#### `get_index_stats`
Get index health check: document count, embedding dimension, and HNSW configuration.

**Parameters:**
- `collection_name` (str, optional) - Collection name (default: `"vectorforge"`)

**Example:**
```
"Get index statistics"
```

---

### **Search**

#### `search_documents`
Semantic search across indexed documents using embeddings. Returns top-k most similar results with optional metadata filtering.

**Parameters:**
- `query` (str) - Search query (natural language)
- `top_k` (int, optional) - Number of results (default: 10, max: 100)
- `source_file` (str, optional) - Filter by source filename (exact match, case-sensitive)
- `chunk_index` (int, optional) - Filter by chunk index
- `collection_name` (str, optional) - Collection name (default: `"vectorforge"`)

**Examples:**
```
"Search for 'machine learning concepts' with top 5 results"

"Search for 'introduction' in file 'textbook.pdf'"

"Find all first chunks containing 'overview'"
```

**Filtering Options:**
- Filter by `source_file` alone → returns all matching chunks from that file
- Filter by `chunk_index` alone → returns all matching chunks at that index (any file)
- Filter by both → returns specific chunk from specific file
- No filters → returns all matching results

---

### **System**

#### `get_metrics`
Get comprehensive metrics: performance stats (query times, percentiles), memory usage, operation counts, uptime, and system info.

**Parameters:**
- `collection_name` (str, optional) - Collection name (default: `"vectorforge"`)

**Example:**
```
"Get system metrics"
```

#### `check_health`
Verify VectorForge API connectivity and status. Returns version and health status.

**Example:**
```
"Check VectorForge health"
```

---

## Configuration

### **MCPConfig Settings**

Located in [`config.py`](config.py):

| Setting | Default | Env Var | Description |
|---------|---------|---------|-------------|
| `SERVER_NAME` | `"VectorForge MCP Server"` | — | Display name reported to MCP clients |
| `SERVER_DESCRIPTION` | `"Model Context Protocol..."` | — | Server description reported to MCP clients |
| `VECTORFORGE_API_BASE_URL` | `"http://localhost:3001"` | `VECTORFORGE_API_BASE_URL` | Base URL of the VectorForge REST API |
| `MCP_HOST` | `"0.0.0.0"` | `MCP_HOST` | Network interface the SSE server binds to |
| `MCP_PORT` | `3002` | `MCP_PORT` | TCP port the SSE server listens on |
| `DEFAULT_COLLECTION_NAME` | `"vectorforge"` | — | Collection used when none is specified |
| `DEFAULT_TOP_K` | `10` | — | Default number of results for `search_documents` |
| `LOG_LEVEL` | `logging.INFO` | — | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | `"%(asctime)s - %(name)s..."` | — | Log message format |

### **Customizing Configuration**

Set environment variables before starting the container, or add them to `docker-compose.yml`:

```yaml
environment:
  - VECTORFORGE_API_BASE_URL=http://vectorforge:3001
  - MCP_PORT=3002
  - MCP_HOST=0.0.0.0
```

For local dev, edit [`config.py`](config.py) class variables directly:

```python
class MCPConfig:
    DEFAULT_COLLECTION_NAME: str = "my-default"
    LOG_LEVEL: int = logging.DEBUG  # Enable debug logging
```

Configuration is validated on server startup via `MCPConfig.validate()`.

---

## Examples

### **Example 1: Index and Search Documents**

**Conversation with Claude:**

```
You: Index this document: "VectorForge is a high-performance vector database"

Claude: I'll add that document to the index.
[Calls add_document tool]
Document indexed with ID: abc-123

You: Search for "vector database"

Claude: I'll search for that.
[Calls search_documents tool]
Found 1 result:
- Score: 0.89
- Content: "VectorForge is a high-performance vector database"
```

### **Example 2: Upload and Search PDF**

```
You: Upload the file /Users/nick/papers/ai_research.pdf

Claude: I'll upload and index that PDF file.
[Calls upload_file tool]
Uploaded ai_research.pdf
- Created 24 chunks
- Generated 24 document IDs

You: What does the paper say about neural networks?

Claude: Let me search for that.
[Calls search_documents with query="neural networks"]
Based on the indexed content, here are the relevant sections...
```

### **Example 3: Search with Filters**

```
You: Search for introductions only in textbook.pdf

Claude: I'll search for introductions specifically in textbook.pdf.
[Calls search_documents with query="introduction", source_file="textbook.pdf"]
Found 3 results from textbook.pdf:
- Chapter 1 Introduction (Score: 0.92)
- Chapter 5 Introduction (Score: 0.85)
- Appendix Introduction (Score: 0.78)

You: Now find all first chunks that mention "overview"

Claude: I'll search for "overview" in all first chunks.
[Calls search_documents with query="overview", chunk_index=0]
Found 2 results:
- guide.pdf, chunk 0 (Score: 0.88)
- manual.pdf, chunk 0 (Score: 0.82)
```

### **Example 4: Multi-Collection Setup**

```
You: Create a collection called "research" for academic papers

Claude: I'll create that collection.
[Calls create_collection with collection_name="research"]
Created collection 'research'

You: Upload /path/to/paper.pdf to the research collection

Claude: I'll upload that to the research collection.
[Calls upload_file with file_path="/path/to/paper.pdf", collection_name="research"]
Uploaded paper.pdf to 'research' - 18 chunks created

You: Search for "neural networks" in research papers only

Claude: [Calls search_documents with query="neural networks", collection_name="research"]
Found relevant sections from your research collection...
```

### **Example 5: Monitoring and Maintenance**

```
You: Check the health of VectorForge

Claude: [Calls check_health tool]
VectorForge API is healthy
- Version: 1.0.3
- Status: healthy

You: Get the current metrics

Claude: [Calls get_metrics tool]
Current VectorForge metrics:
- Total documents: 150
- Total queries: 1,247
- Average query time: 12.3ms
- Memory usage: 45.2 MB
- Uptime: 3 days, 5 hours
```

---

## Troubleshooting

### **Server Won't Start**

**Error:** `ModuleNotFoundError: No module named 'fastmcp'`

**Solution:**
```bash
# Install MCP dependencies
uv sync --group mcp
```

---

### **Connection Refused**

**Error:** `VectorForge API is not available - Connection refused`

**Cause:** VectorForge API is not running.

**Solution:**
```bash
docker compose up -d
```

---

### **mcp-remote Not Found / npx Fails**

**Error:** `npx: command not found` or `mcp-remote` fails to launch

**Cause:** Node.js is not installed on the host machine.

**Solution:** Install Node.js from [nodejs.org](https://nodejs.org/) (LTS version recommended), then restart Claude Desktop.

---

### **Tools Not Appearing in Claude**

**Check:**
1. Server is running (`vectorforge-mcp` command executed)
2. `claude_desktop_config.json` is correctly configured
3. Claude Desktop has been restarted after config changes
4. Check Claude Desktop logs for connection errors

**Verify config location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%/Claude/claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

---

### **Enable Debug Logging**

Edit [`config.py`](config.py):
```python
LOG_LEVEL: int = logging.DEBUG
```

---

## Architecture Details

### **Error Handler Decorator**

All tools use `@handle_tool_errors` from [`decorators.py`](decorators.py):

```python
@handle_tool_errors
def my_tool(param: str) -> dict:
    data = get("/some/endpoint")
    return build_success_response(data)
```

This automatically catches and formats:
- `requests.ConnectionError` → "API not available"
- `requests.Timeout` → "Request timeout"
- `requests.HTTPError` → HTTP status code and response body
- Generic exceptions → "Operation failed"

### **Response Format**

All tools return standardized responses:

**Success:**
```json
{
  "success": true,
  "data": { /* API response */ }
}
```

**Error:**
```json
{
  "success": false,
  "error": "Error message",
  "details": "Additional context"
}
```

---

## Related Projects

- **VectorForge Core** - Main vector database ([README](../README.md))
- **FastMCP** - MCP server framework ([GitHub](https://github.com/jlowin/fastmcp))
- **Model Context Protocol** - Protocol specification ([Anthropic](https://modelcontextprotocol.io/))

---

<div align="center">
  <strong>VectorForge MCP Server - Semantic Search for AI Assistants</strong>
</div>
