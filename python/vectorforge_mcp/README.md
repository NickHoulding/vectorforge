# VectorForge MCP Server

![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)
![FastMCP](https://img.shields.io/badge/FastMCP-2.14+-green.svg)
![Version](https://img.shields.io/badge/version-0.9.0-orange.svg)

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

- Index and search documents semantically
- Upload and process files (PDF, TXT)
- Manage vector embeddings
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
- **Python 3.11+** - Modern Python with async support
- **FastMCP 2.14+** - Fast, modern MCP server framework
- **VectorForge API** - Core vector database (required)

### **Dependencies**
- **httpx** - Async HTTP client for API communication
- **FastAPI** - Used by VectorForge API (indirect)
- **Pydantic** - Data validation for VectorForge models

---

## Key Features

### **13 MCP Tools**
Organized into 5 categories:

**Documents** (3 tools)
- Get document by ID
- Add document with metadata
- Delete document

**Files** (3 tools)
- List indexed files
- Upload and process files
- Delete all chunks from a file

**Index** (4 tools)
- Get index statistics
- Build/rebuild index
- Save index to disk
- Load index from disk

**Search** (1 tool)
- Semantic search with top-k results

**System** (2 tools)
- Get comprehensive metrics
- Health check

### **Production-Ready Features**
- **Comprehensive Error Handling** - Catches network errors, timeouts, API version mismatches
- **Async Support** - Upload operations run asynchronously
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
│ • 13 MCP Tools  │
│ • Error Handler │
│ • Logging       │
└────────┬────────┘
         │ HTTP/REST
         │
┌────────▼────────┐
│  VectorForge    │
│  REST API       │ (Required - must be running)
├─────────────────┤
│ • Vector Engine │
│ • Embeddings    │
│ • Storage       │
└─────────────────┘
```

### **Module Structure**

```
vectorforge_mcp/
├── __init__.py
├── config.py          # Configuration settings
├── decorators.py      # Error handling decorator
├── instance.py        # FastMCP server instance
├── server.py          # Entry point and main()
├── utils.py           # Response builders
└── tools/             # MCP tool implementations
    ├── documents.py   # Document management tools
    ├── files.py       # File upload/management tools
    ├── index.py       # Index management tools
    ├── search.py      # Semantic search tool
    └── system.py      # System info tools
```

### **Error Handling Strategy**

All tools are wrapped with `@handle_api_errors` decorator that catches:
- **Connection Errors** - API not running, network issues
- **Timeouts** - Request timeouts with helpful messages
- **Version Mismatches** - HTTP 422 detected and reported
- **Generic Exceptions** - Fallback error handling

This provides consistent error responses across all tools without duplicate code.

---

## Getting Started

### **Prerequisites**

1. **VectorForge API must be running**
   ```bash
   # In one terminal, start VectorForge API
   cd vectorforge/python
   uv run vectorforge-api
   ```

2. **Python 3.11+** installed
3. **uv** package manager

### **Installation**

```bash
# Navigate to project root
cd vectorforge/python

# Install with MCP dependencies
uv sync --group mcp

# Or using pip
pip install -e ".[mcp]"
```

### **Verify Installation**

```bash
# Check if command is available
which vectorforge-mcp

# Test configuration validation
python -c "from vectorforge_mcp.config import MCPConfig; MCPConfig.validate(); print('✓ Config valid')"
```

### **Running the Server**

#### **Option 1: Using Console Script (Recommended)**
```bash
vectorforge-mcp
```

#### **Option 2: Using Python Module**
```bash
python -m vectorforge_mcp.server
```

#### **Option 3: Direct Execution**
```bash
cd python/vectorforge_mcp
python server.py
```

The server will:
1. Validate configuration
2. Configure logging (INFO level by default)
3. Start listening for MCP client connections

---

## Usage

### **Connecting from Claude Desktop**

1. **Configure Claude Desktop**

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or equivalent:

```json
{
  "mcpServers": {
    "vectorforge": {
      "command": "vectorforge-mcp",
      "env": {
        "PYTHONPATH": "/path/to/vectorforge/python"
      }
    }
  }
}
```

2. **Restart Claude Desktop**

3. **Verify Connection**

In Claude, ask: "What VectorForge tools are available?"

Claude should list all 13 MCP tools.

### **Connecting from Other MCP Clients**

The server uses stdio transport and can connect to any MCP-compatible client. Refer to your client's documentation for connection setup.

---

## Available Tools

### **Documents**

#### `get_document`
Fetch document content and metadata by ID.

**Parameters:**
- `doc_id` (str) - Unique document identifier (UUID)

**Example:**
```
"Get document 550e8400-e29b-41d4-a716-446655440000"
```

#### `add_document`
Index text content for semantic search. Generates embeddings automatically.

**Parameters:**
- `content` (str) - Document text content (required, non-empty)
- `metadata` (dict, optional) - Custom metadata

**Example:**
```
"Add a document with content 'Python is a programming language' and metadata {\"topic\": \"tech\"}"
```

#### `delete_document`
Permanently remove a document and its embeddings.

**Parameters:**
- `doc_id` (str) - Document ID to delete

**Example:**
```
"Delete document 550e8400-e29b-41d4-a716-446655440000"
```

---

### **Files**

#### `list_files`
Get all filenames that have been uploaded and chunked.

**Example:**
```
"List all indexed files"
```

#### `upload_file`
Upload and index a file (PDF, TXT). Extracts text, chunks it, generates embeddings.

**Parameters:**
- `file_path` (str) - Absolute path to file

**Example:**
```
"Upload file /path/to/document.pdf"
```

#### `delete_file`
Delete all document chunks from a specific uploaded file.

**Parameters:**
- `filename` (str) - Name of the source file

**Example:**
```
"Delete file document.pdf"
```

---

### **Index**

#### `get_index_stats`
Get lightweight index health check: document counts, embedding dimension, deletion ratio, compaction status.

**Example:**
```
"Get index statistics"
```

#### `build_index`
Rebuild entire vector index from scratch. Regenerates all embeddings.

**Example:**
```
"Rebuild the index"
```

#### `save_index`
Persist index to disk (embeddings + metadata). Enables fast recovery.

**Parameters:**
- `directory` (str, optional) - Directory path (default: './data')

**Example:**
```
"Save the index to disk"
```

#### `load_index`
Restore index from disk. Loads previously saved embeddings and metadata.

**Parameters:**
- `directory` (str, optional) - Directory path (default: './data')

**Example:**
```
"Load the index from disk"
```

---

### **Search**

#### `search_documents`
Semantic search across indexed documents using embeddings. Returns top-k most similar results.

**Parameters:**
- `query` (str) - Search query (natural language)
- `top_k` (int, optional) - Number of results (default: 10, max: 100)

**Example:**
```
"Search for 'machine learning concepts' with top 5 results"
```

---

### **System**

#### `get_metrics`
Get comprehensive metrics: performance stats (query times, percentiles), memory usage, operation counts, uptime, and system info.

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

| Setting | Default | Description |
|---------|---------|-------------|
| `SERVER_NAME` | `"VectorForge MCP Server"` | Name shown to MCP clients |
| `SERVER_DESCRIPTION` | `"Model Context Protocol..."` | Server description |
| `VECTORFORGE_API_BASE_URL` | `"http://localhost:3001"` | VectorForge API URL |
| `LOG_LEVEL` | `logging.INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | `"%(asctime)s - %(name)s..."` | Log message format |

### **Customizing Configuration**

Edit [`config.py`](config.py) and modify class variables:

```python
class MCPConfig:
    SERVER_NAME: str = "My Custom VectorForge Server"
    LOG_LEVEL: int = logging.DEBUG  # Enable debug logging
    VECTORFORGE_API_BASE_URL: str = "http://custom-host:8080"
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
✓ Document indexed with ID: abc-123

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
✓ Uploaded ai_research.pdf
- Created 24 chunks
- Generated 24 document IDs

You: What does the paper say about neural networks?

Claude: Let me search for that.
[Calls search_documents with query="neural networks"]
Based on the indexed content, here are the relevant sections...
```

### **Example 3: Monitoring and Maintenance**

```
You: Check the health of VectorForge

Claude: [Calls check_health tool]
✓ VectorForge API is healthy
- Version: 0.9.0
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
# Start VectorForge API in another terminal
vectorforge-api
```

---

### **Command Not Found**

**Error:** `vectorforge-mcp: command not found`

**Cause:** Package not installed with entry points enabled.

**Solution:**
```bash
# Ensure package is installed
uv sync
# Or reinstall
pip install -e .
```

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

Or set environment variable:
```bash
export LOG_LEVEL=DEBUG
vectorforge-mcp
```

---

## Architecture Details

### **Error Handler Decorator**

All tools use `@handle_api_errors` from [`decorators.py`](decorators.py):

```python
@handle_api_errors
def my_tool(param: str) -> dict:
    response = api.some_operation(param)
    return build_success_response(response)
```

This automatically catches and formats:
- `ConnectionRefusedError` → "API not available"
- `TimeoutError` → "Request timeout"
- `HTTPException(422)` → "API version mismatch"
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

- **VectorForge Core** - Main vector database ([README](../../README.md))
- **FastMCP** - MCP server framework ([GitHub](https://github.com/jlowin/fastmcp))
- **Model Context Protocol** - Protocol specification ([Anthropic](https://modelcontextprotocol.io/))

---

## Contributing

Contributions to the MCP server are welcome! Please see the main [VectorForge README](../../README.md) for contribution guidelines.

### **Testing**

```bash
# Run MCP-specific tests
pytest tests/test_mcp_tools.py -v

# Test with coverage
pytest tests/test_mcp_tools.py --cov=vectorforge_mcp
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](../../LICENSE) file for details.

---

<div align="center">
  <strong>VectorForge MCP Server - Semantic Search for AI Assistants</strong>
</div>
