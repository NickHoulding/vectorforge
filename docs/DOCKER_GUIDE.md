# VectorForge Docker Implementation Guide

**A Beginner-Friendly Tutorial for Containerizing VectorForge**

---

## 📚 Table of Contents

1. [Docker Basics - What You Need to Know](#docker-basics)
2. [Phase 1: Prepare Your Code for Docker](#phase-1-prepare-code)
3. [Phase 2: Create Your First Dockerfile](#phase-2-dockerfile)
4. [Phase 3: Build and Run Your Container](#phase-3-build-and-run)
5. [Phase 4: Add Docker Compose](#phase-4-docker-compose)
6. [Phase 5: Test Data Persistence](#phase-5-data-persistence)
7. [Troubleshooting Common Issues](#troubleshooting)
8. [Next Steps](#next-steps)

---

## Docker Basics - What You Need to Know

### What is Docker?

Docker packages your application and all its dependencies into a **container** - a lightweight, standalone, executable package. Think of it like shipping your app in a standardized box that runs the same way everywhere.

**Key Concepts:**

1. **Image**: A blueprint/template for your container (like a class in programming)
2. **Container**: A running instance of an image (like an object/instance)
3. **Dockerfile**: Instructions for building an image (like a recipe)
4. **Volume**: Persistent storage that survives container restarts
5. **Port Mapping**: Connecting container ports to your host machine

**Why Docker for VectorForge?**
- ✅ Consistent environment across dev/prod
- ✅ Easy deployment (one command)
- ✅ Isolated dependencies (no conflicts)
- ✅ Easy to share and reproduce

### Prerequisites

```bash
# Check if Docker is installed
docker --version

# Check if Docker Compose is installed
docker compose version

# If not installed, visit: https://docs.docker.com/get-docker/
```

---

## Phase 1: Prepare Code for Docker

**Goal**: Make VectorForge configurable via environment variables instead of hardcoded values.

**Why?** Docker best practice is to configure apps at runtime using environment variables, not by editing code.

### 1.1: Add Environment Variable Support to Config

**What we're doing:** Allowing configuration values to come from environment variables with sensible defaults.

**File to edit:** `python/vectorforge/config.py`

**Find this section** (around line 96-101):
```python
# =============================================================================
# ChromaDB Configuration
# =============================================================================

CHROMA_PERSIST_DIR: str = "chroma_data"
"""Directory name for ChromaDB persistent storage (relative to vector_engine.py)."""
```

**Replace with:**
```python
# =============================================================================
# ChromaDB Configuration
# =============================================================================

# Smart default: /data/chroma for containers, ./data/chroma for local dev
_default_chroma = "/data/chroma" if os.path.exists("/data") else "./data/chroma"
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_DATA_DIR", _default_chroma)
"""Directory for ChromaDB persistent storage. Configurable via CHROMA_DATA_DIR env var."""

MODEL_CACHE_DIR: str = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
"""Directory for HuggingFace model cache. Configurable via HF_HOME env var."""
```

**Docker Concept Learned:** 
- **Environment Variables**: Docker containers can receive configuration through `ENV` instructions or `-e` flags
- **`os.getenv(key, default)`**: Python function that reads environment variables with fallback defaults
- **Smart Defaults**: Code can detect environment (Docker vs local) and adjust behavior

**Why this matters for Docker:**
- Same Docker image can be configured differently without rebuilding
- Follows [12-factor app](https://12factor.net/config) principles
- Required for cloud deployments (Kubernetes, AWS ECS, etc.)

**Why the smart default?**
- Local dev: `/data` doesn't exist, uses `./data/chroma` (no permission errors!)
- Docker: `/data` exists, uses `/data/chroma` (volume mount)
- Override: Can always set `CHROMA_DATA_DIR` env var for custom paths

---

**File to edit:** `python/vectorforge/api/config.py`

**Find this section** (around line 12-16):
```python
API_PORT: int = 3001
"""Default port for the FastAPI server."""

API_HOST: str = "0.0.0.0"
"""Default host binding for the FastAPI server."""
```

**Replace with:**
```python
API_PORT: int = int(os.getenv("API_PORT", "3001"))
"""API server port. Configurable via API_PORT env var."""

API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
"""API server host binding. Configurable via API_HOST env var."""

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
"""Logging level (DEBUG, INFO, WARNING, ERROR). Configurable via LOG_LEVEL env var."""
```

**Add import at the top of the file:**
```python
import os
```

---

### 1.2: Fix ChromaDB Data Path (Critical for Docker)

**What we're doing:** Making ChromaDB store data in a configurable location instead of inside the source code directory.

**Why?** In Docker:
- Source code directories are read-only or ephemeral
- Data must be stored in **volumes** (persistent storage)
- `/data` is a common convention for application data

**File to edit:** `python/vectorforge/vector_engine.py`

**Find this code** (around line 134-139):
```python
def __init__(self) -> None:
    """Initialize the VectorEngine with ChromaDB backend.
    
    Creates a vector database using ChromaDB for storage and retrieval,
    with the 'all-MiniLM-L6-v2' sentence transformer model for embedding
    generation. Initializes metrics tracking.
    """
    engine_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_path = os.path.join(engine_dir, VFGConfig.CHROMA_PERSIST_DIR)
```

**Replace with:**
```python
def __init__(self) -> None:
    """Initialize the VectorEngine with ChromaDB backend.
    
    Creates a vector database using ChromaDB for storage and retrieval,
    with the 'all-MiniLM-L6-v2' sentence transformer model for embedding
    generation. Initializes metrics tracking.
    """
    # Use absolute path from config (supports Docker volumes)
    chroma_path = VFGConfig.CHROMA_PERSIST_DIR
    
    # Convert relative paths to absolute (for backward compatibility)
    if not os.path.isabs(chroma_path):
        chroma_path = os.path.abspath(chroma_path)
    
    # Ensure directory exists
    os.makedirs(chroma_path, exist_ok=True)
```

**Docker Concept Learned:**
- **Volumes**: Docker volumes are mounted at specific paths (like `/data`)
- **Absolute vs Relative Paths**: Docker containers need absolute paths for volumes
- **`os.makedirs(exist_ok=True)`**: Creates directory if it doesn't exist (safe for Docker startup)

---

### 1.3: Add Basic Logging Configuration

**What we're doing:** Making logs configurable so you can debug Docker containers easily.

**Create new file:** `python/vectorforge/logging_config.py`

```python
"""Logging configuration for VectorForge"""

import logging
import sys

from vectorforge.api.config import APIConfig


def configure_logging() -> None:
    """Configure logging based on LOG_LEVEL environment variable.
    
    Sets up basic logging to stdout (required for Docker log collection).
    Respects LOG_LEVEL env var: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    log_level = APIConfig.LOG_LEVEL
    
    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level not in valid_levels:
        log_level = "INFO"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,  # Important: Docker reads from stdout
        force=True,  # Override any existing config
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at level: {log_level}")
```

**Docker Concept Learned:**
- **Container Logs**: Docker captures stdout/stderr from containers
- **`docker logs <container>`**: Command to view container output
- **Why stdout?**: Docker log drivers (journald, fluentd, etc.) read from stdout

---

**File to edit:** `python/vectorforge/api/__init__.py`

**Add this import at the top:**
```python
from vectorforge.logging_config import configure_logging
```

**Add this line right after the imports (around line 13):**
```python
# Configure logging before creating engine
configure_logging()
```

Your `__init__.py` should now look like:
```python
"""VectorForge API Initialization"""

from fastapi import FastAPI

from vectorforge import __version__
from vectorforge.logging_config import configure_logging
from vectorforge.vector_engine import VectorEngine

# Configure logging before creating engine
configure_logging()

app: FastAPI = FastAPI(
    title="VectorForge API",
    version=__version__,
    description="High-performance in-memory vector database with semantic search",
)
engine: VectorEngine = VectorEngine()

from vectorforge.api import documents, files, index, search, system

app.include_router(documents.router)
app.include_router(files.router)
app.include_router(index.router)
app.include_router(search.router)
app.include_router(system.router)

__all__: list[str] = ["app", "engine"]
```

---

### 1.4: Add Enhanced Health Checks (for Docker)

**What we're doing:** Adding readiness and liveness probes that Docker/Kubernetes can use to monitor container health.

**Docker Concept:**
- **Health Checks**: Docker can automatically restart unhealthy containers
- **Readiness Probe**: "Is the app ready to receive traffic?"
- **Liveness Probe**: "Is the app still running?"

**File to edit:** `python/vectorforge/api/system.py`

**Add these new endpoints after the existing `/health` endpoint:**

```python
@router.get("/health/ready")
async def readiness_check() -> dict[str, str]:
    """Readiness probe for container orchestration.
    
    Checks if VectorForge is fully initialized and ready to handle requests.
    Returns 200 if ready, 503 if not ready.
    
    Used by Docker/Kubernetes to know when to send traffic to this container.
    """
    try:
        # Verify ChromaDB is accessible
        doc_count = engine.collection.count()
        
        # Verify model is loaded
        if engine.model is None:
            raise RuntimeError("Model not loaded")
        
        return {
            "status": "ready",
            "documents": doc_count,
            "model": engine.model_name,
        }
    except Exception as e:
        # Return 503 Service Unavailable
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}"
        )


@router.get("/health/live")
async def liveness_check() -> dict[str, str]:
    """Liveness probe for container orchestration.
    
    Simple check that the API is responding.
    Returns 200 if alive, used by Docker/Kubernetes to detect hung processes.
    """
    return {"status": "alive"}
```

**Add import at the top if not already present:**
```python
from fastapi import HTTPException
```

---

### 1.5: Fix Test Fixtures (Prevent Test Pollution)

**What we're doing:** Making tests use temporary directories instead of polluting source code.

**Why?** Tests currently write to `vectorforge/chroma_data/`, which:
- Gets committed to git accidentally
- Interferes with Docker builds
- Causes flaky tests

**File to edit:** `python/tests/conftest.py`

**Add these imports at the top:**
```python
import tempfile
```

**Add this new fixture after the `anyio_backend` fixture:**
```python
@pytest.fixture(autouse=True)
def use_temp_chroma_dir(monkeypatch: pytest.MonkeyPatch) -> Generator[str, Any, None]:
    """Use temporary directory for ChromaDB in tests.
    
    Automatically applied to all tests. Ensures tests don't pollute
    the source code directory with test data.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        chroma_path = os.path.join(tmpdir, "chroma_test")
        monkeypatch.setenv("CHROMA_DATA_DIR", chroma_path)
        yield chroma_path
```

**Add import:**
```python
import os
```

**Update the type hint imports at the top:**
```python
from typing import Any, Generator
```

---

### 1.6: Fix Entry Point Configuration

**What we're doing:** Fixing the command-line entry point so `vectorforge-api` command works correctly.

**Why?** The `pyproject.toml` currently points to the wrong location for the `main()` function, causing an import error.

**The Problem:**
```bash
$ uv run vectorforge-api
ImportError: cannot import name 'main' from 'vectorforge.api'
```

**File to edit:** `python/pyproject.toml`

**Find this line** (around line 30):
```toml
[project.scripts]
vectorforge-api = "vectorforge.api:main"
vectorforge-mcp = "vectorforge_mcp.server:main"
```

**Replace with:**
```toml
[project.scripts]
vectorforge-api = "vectorforge.__main__:main"
vectorforge-mcp = "vectorforge_mcp.server:main"
```

**What changed:**
- **Before**: `vectorforge.api:main` (main function doesn't exist there!)
- **After**: `vectorforge.__main__:main` (main function actually lives here)

**After making this change, reinstall the package scripts:**
```bash
cd python
uv sync
```

**Docker Concept Learned:**
- **Entry Points**: Python packages can define command-line scripts
- **Format**: `command-name = "module.path:function_name"`
- **Why sync?**: Changes to `pyproject.toml` scripts require reinstalling the package

---

### 1.7: Verify Code Changes Work Locally

**Before moving to Docker, let's test locally:**

```bash
cd python

# Test with default config (should work as before)
uv run vectorforge-api &
API_PID=$!
sleep 10
curl http://localhost:3001/health
kill $API_PID

# Test with custom config via environment variables
export CHROMA_DATA_DIR="/tmp/vectorforge_test"
export LOG_LEVEL="DEBUG"
export API_PORT="3002"

uv run vectorforge-api &
API_PID=$!
sleep 10
curl http://localhost:3002/health/ready
kill $API_PID

# Run tests to ensure nothing broke
uv run pytest tests/ -v

# Clean up
unset CHROMA_DATA_DIR LOG_LEVEL API_PORT
```

**Expected Results:**
- ✅ API starts successfully
- ✅ Health endpoints return 200 OK
- ✅ Tests pass
- ✅ ChromaDB data created in `/tmp/vectorforge_test/` (not in source code)

**If tests fail:** Review the changes carefully, ensure imports are correct.

---

## Phase 2: Create Your First Dockerfile

**Goal**: Write the instructions for building a Docker image of VectorForge.

### What is a Dockerfile?

A `Dockerfile` is like a recipe that tells Docker how to build your application image. Each instruction creates a **layer** in the image.

**Common Dockerfile Instructions:**
- `FROM`: Base image to start from (e.g., Python 3.11)
- `WORKDIR`: Set working directory inside container
- `COPY`: Copy files from your computer into the image
- `RUN`: Execute commands during build (e.g., install packages)
- `ENV`: Set environment variables
- `EXPOSE`: Document which port the app uses
- `CMD`: Command to run when container starts

### 2.1: Create .dockerignore (Important!)

**What it does:** Tells Docker which files to ignore when copying (like `.gitignore`).

**Why?** Prevents:
- Copying unnecessary files (node_modules, test data, etc.)
- Huge image sizes
- Leaking secrets

**Create file:** `python/.dockerignore`

```
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
*.egg
*.egg-info/
dist/
build/

# Virtual environments
.venv/
venv/
env/

# Testing
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/

# Data directories (we'll use volumes for these)
chroma_data/
data/
*.db
*.sqlite

# Development
.git/
.gitignore
.pre-commit-config.yaml
tests/
benchmarks/

# Documentation
*.md
docs/

# IDE
.vscode/
.idea/
*.swp
*.swo
```

**Docker Concept Learned:**
- **Build Context**: Docker sends all files in the directory to the build process
- **.dockerignore**: Reduces build context size, speeds up builds, prevents leaking secrets

---

### 2.2: Create the Dockerfile

**Create file:** `python/Dockerfile`

Let's build it step by step with explanations:

```dockerfile
# ============================================================================
# Stage 1: Base Image
# ============================================================================
# Start from official Python 3.11 slim image (Debian-based, minimal size)
FROM python:3.11-slim AS base

# Why python:3.11-slim?
# - Official image (trusted, maintained)
# - "slim" = smaller size (~150MB vs ~900MB for full Python)
# - Includes pip and common system libraries
# - Based on Debian (familiar, stable)

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# PYTHONUNBUFFERED=1: Forces Python to output logs immediately (important for Docker logs)
# PYTHONDONTWRITEBYTECODE=1: Don't create .pyc files (smaller image)
# PIP_NO_CACHE_DIR=1: Don't cache pip downloads (smaller image)

# Install system dependencies required by some Python packages
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# curl: Needed for health checks
# rm -rf /var/lib/apt/lists/*: Clean up package lists (smaller image)


# ============================================================================
# Stage 2: Builder (Install Dependencies)
# ============================================================================
FROM base AS builder

# Why a separate builder stage?
# - Keeps final image smaller (we don't need build tools in production)
# - Multi-stage builds = best practice

# Install uv (fast Python package installer)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# COPY --from: Copies files from another image (official uv image)
# This is better than curl | sh installs (reproducible, cached)

# Set working directory
WORKDIR /app

# Copy only dependency files first (for better caching)
COPY pyproject.toml uv.lock ./

# Why copy these separately?
# - Docker caches each layer
# - If code changes but dependencies don't, Docker reuses cached layer
# - Speeds up rebuilds significantly

# Copy source code (required for building the local package)
COPY vectorforge/ ./vectorforge/
COPY vectorforge_mcp/ ./vectorforge_mcp/

# Why copy source code here?
# - pyproject.toml has [tool.uv] package = true
# - This means uv sync needs to build the vectorforge package
# - Building requires the source code to be present
# - Without this, build fails with "Unable to determine which files to ship"

# Install Python dependencies (includes building the local package)
RUN uv sync --frozen --no-dev

# --frozen: Use exact versions from uv.lock (reproducible builds)
# --no-dev: Skip development dependencies (smaller image)
# This also installs vectorforge package in editable mode (creates .pth file)

# Pre-download the sentence transformer model (bake into image)
RUN uv run python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

# Why bake model into image?
# - Fast startup (no download on first run)
# - Consistent across deployments
# - Trade-off: Larger image (~800MB vs ~200MB)


# ============================================================================
# Stage 3: Final Production Image
# ============================================================================
FROM base AS production

# Start fresh from base image (without build tools)

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy HuggingFace model cache from builder stage
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Why copy from builder?
# - Final image doesn't have uv or build tools
# - Only runtime dependencies
# - Smaller, more secure image

# Copy application code (IMPORTANT: Must copy to production stage too!)
COPY vectorforge/ ./vectorforge/
COPY vectorforge_mcp/ ./vectorforge_mcp/

# Why copy source code AGAIN in production stage?
# - The builder stage installed the package in "editable mode"
# - This creates a .pth file that points to /app/vectorforge
# - The .pth file is copied with .venv, but the source code is NOT
# - Without source code here, Python can't import vectorforge module
# - Result: "ModuleNotFoundError: No module named 'vectorforge'"
# - Solution: Copy source to both builder AND production stages

# Create data directory (where ChromaDB will store data)
RUN mkdir -p /data/chroma

# Set environment variables (defaults that can be overridden)
ENV CHROMA_DATA_DIR=/data/chroma \
    HF_HOME=/root/.cache/huggingface \
    API_PORT=3001 \
    API_HOST=0.0.0.0 \
    LOG_LEVEL=INFO \
    PATH="/app/.venv/bin:$PATH"

# PATH="/app/.venv/bin:$PATH": Adds venv to PATH so we can run commands

# Expose port 3001 (documentation, doesn't actually open port)
EXPOSE 3001

# EXPOSE: Documents which port the app uses
# Note: Doesn't actually publish the port, use -p flag when running

# Add health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:3001/health/live || exit 1

# HEALTHCHECK: Docker monitors container health
# --interval=30s: Check every 30 seconds
# --timeout=5s: Fail if check takes >5 seconds
# --start-period=40s: Grace period for app startup
# --retries=3: Mark unhealthy after 3 consecutive failures
# exit 1: Signals unhealthy status

# Set non-root user for security (optional but recommended)
# RUN useradd -m -u 1000 vectorforge && \
#     chown -R vectorforge:vectorforge /app /data
# USER vectorforge

# Why non-root?
# - Security best practice (principle of least privilege)
# - Commented out for now (can enable later if needed)

# Run the API server
CMD ["vectorforge-api"]

# CMD: Default command when container starts
# Uses the entrypoint defined in pyproject.toml
```

**Save this file as:** `python/Dockerfile`

---

### 2.3: Understanding the Dockerfile Structure

**Multi-Stage Build Benefits:**

```
┌─────────────────┐
│  base           │  Base Python 3.11 image
│  (150 MB)       │
└────────┬────────┘
         │
         ├─────────────────┐
         │                 │
┌────────▼────────┐  ┌─────▼──────────┐
│  builder        │  │  production    │
│  + uv           │  │  (final image) │
│  + build tools  │  │                │
│  + dependencies │  │  Copies only:  │
│  + model DL     │  │  - .venv       │
│  (1.2 GB)       │  │  - model cache │
└─────────────────┘  │  - app code    │
                     │  (800 MB)      │
                     └────────────────┘
```

**Why this matters:**
- Final image is smaller (no build tools)
- Faster downloads/deploys
- More secure (fewer packages = smaller attack surface)

---

### 2.4: Create docker-compose.yml (Easy Container Management)

**What is Docker Compose?**
- Tool for defining multi-container applications
- Uses YAML file instead of long `docker run` commands
- Manages volumes, networks, environment variables
- One command to start everything: `docker compose up`

**Create file:** `python/docker-compose.yml`

```yaml
# Docker Compose file for VectorForge
# Usage: docker compose up
version: '3.8'

services:
  # Define the VectorForge service
  vectorforge:
    # Build configuration
    build:
      context: .
      dockerfile: Dockerfile
      # target: production  # Use this stage from multi-stage build
    
    # Image name after building
    image: vectorforge:latest
    
    # Container name (easier to reference)
    container_name: vectorforge
    
    # Port mapping: host:container
    # Access the API at http://localhost:3001
    ports:
      - "3001:3001"
    
    # Environment variables (override defaults)
    environment:
      - CHROMA_DATA_DIR=/data/chroma
      - LOG_LEVEL=INFO
      # Uncomment to customize:
      # - API_PORT=3001
      # - API_HOST=0.0.0.0
    
    # Volume mounts (persistent storage)
    volumes:
      - vectorforge-data:/data
    
    # Health check configuration
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3001/health/live"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 40s
    
    # Restart policy
    restart: unless-stopped
    
    # Resource limits (optional, uncomment to use)
    # deploy:
    #   resources:
    #     limits:
    #       cpus: '2'
    #       memory: 2G
    #     reservations:
    #       cpus: '1'
    #       memory: 1G

# Named volumes (persistent storage)
volumes:
  vectorforge-data:
    driver: local
    # Data persists even if container is deleted
```

**Docker Compose Concepts Learned:**

1. **Services**: Containers to run (we have one: `vectorforge`)
2. **Ports**: `"3001:3001"` means "map host port 3001 to container port 3001"
3. **Volumes**: Named storage that persists data
4. **Environment**: Override default env vars
5. **Restart Policy**:
   - `no`: Never restart
   - `always`: Always restart
   - `unless-stopped`: Restart unless manually stopped
   - `on-failure`: Only restart on error

**Volume Types:**
- **Named Volume** (`vectorforge-data`): Managed by Docker, best for production
- **Bind Mount** (`./data:/data`): Maps host directory to container, good for development
- **Anonymous Volume**: Unnamed, harder to manage

---

## Phase 3: Build and Run Your Container

**Goal**: Build the Docker image and run your first container!

### 3.1: Build the Docker Image

```bash
cd python

# Build the image (this will take a few minutes first time)
docker build -t vectorforge:latest .

# -t vectorforge:latest: Tag the image with name "vectorforge" and tag "latest"
# . : Build context (current directory)
```

**What's happening:**
1. Docker reads the Dockerfile
2. Executes each instruction in order
3. Creates layers (cached for future builds)
4. Downloads base images and dependencies
5. Pre-downloads the AI model (~90MB)
6. Tags the final image

**Expected output:**
```
[+] Building 180.5s (15/15) FINISHED
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 2.3kB
 => [internal] load .dockerignore
 => [base 1/3] FROM docker.io/library/python:3.11-slim
 => [builder 1/5] COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
 => [builder 2/5] WORKDIR /app
 => [builder 3/5] COPY pyproject.toml uv.lock ./
 => [builder 4/5] RUN uv sync --frozen --no-dev
 => [builder 5/5] RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
 => [production 1/4] COPY --from=builder /app/.venv /app/.venv
 => [production 2/4] COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
 => [production 3/4] COPY vectorforge/ ./vectorforge/
 => [production 4/4] RUN mkdir -p /data/chroma
 => exporting to image
 => => naming to docker.io/library/vectorforge:latest
```

**Verify the image was created:**
```bash
docker images | grep vectorforge

# Expected output:
# vectorforge   latest   abc123def456   2 minutes ago   850MB
```

---

### 3.2: Run Your First Container

**Simple run (foreground):**
```bash
docker run --rm -p 3001:3001 vectorforge:latest

# --rm: Remove container when it stops (cleanup)
# -p 3001:3001: Map port 3001 on host to port 3001 in container
# vectorforge:latest: Image name and tag
```

**What you'll see:**
```
2024-02-23 15:30:00 - vectorforge.logging_config - INFO - Logging configured at level: INFO
2024-02-23 15:30:01 - uvicorn - INFO - Started server process [1]
2024-02-23 15:30:01 - uvicorn - INFO - Waiting for application startup.
2024-02-23 15:30:01 - uvicorn - INFO - Application startup complete.
2024-02-23 15:30:01 - uvicorn - INFO - Uvicorn running on http://0.0.0.0:3001
```

**Test it works (open a new terminal):**
```bash
curl http://localhost:3001/health

# Expected: {"status":"healthy","version":"0.9.0"}

curl http://localhost:3001/health/ready

# Expected: {"status":"ready","documents":0,"model":"all-MiniLM-L6-v2"}
```

**Stop the container:** Press `Ctrl+C` in the terminal where it's running.

---

### 3.3: Run Container in Background (Detached Mode)

```bash
# Run in background
docker run -d \
  --name vectorforge \
  -p 3001:3001 \
  vectorforge:latest

# -d: Detached mode (runs in background)
# --name vectorforge: Give container a friendly name
```

**Useful Docker commands:**

```bash
# List running containers
docker ps

# View container logs
docker logs vectorforge

# Follow logs in real-time (like tail -f)
docker logs -f vectorforge

# Stop the container
docker stop vectorforge

# Start it again
docker start vectorforge

# Remove the container
docker rm vectorforge

# Remove container even if running (force)
docker rm -f vectorforge
```

---

### 3.4: Run with Custom Configuration

**Example: Change port and log level**

```bash
docker run -d \
  --name vectorforge \
  -p 8080:8080 \
  -e API_PORT=8080 \
  -e LOG_LEVEL=DEBUG \
  vectorforge:latest

# -p 8080:8080: Map host port 8080 to container port 8080
# -e API_PORT=8080: Set environment variable
# -e LOG_LEVEL=DEBUG: Enable debug logging

# Test it:
curl http://localhost:8080/health
```

**Docker Concepts Learned:**
- **Port Mapping**: `-p HOST:CONTAINER`
- **Environment Variables**: `-e KEY=VALUE`
- **Detached Mode**: `-d` runs in background
- **Container Names**: `--name` for easy reference

---

### 3.5: Run with Docker Compose (Recommended)

**Much easier than long docker run commands!**

```bash
cd python

# Start all services defined in docker-compose.yml
docker compose up -d

# -d: Detached mode (background)
```

**What's happening:**
1. Docker Compose reads `docker-compose.yml`
2. Creates the `vectorforge-data` volume if it doesn't exist
3. Builds the image if not already built
4. Starts the container with all configurations

**Useful Docker Compose commands:**

```bash
# View logs
docker compose logs

# Follow logs
docker compose logs -f

# View container status
docker compose ps

# Stop all services (containers keep existing)
docker compose stop

# Start stopped services
docker compose start

# Stop and remove containers (volumes persist)
docker compose down

# Stop, remove containers AND volumes (deletes data!)
docker compose down -v

# Rebuild image and restart
docker compose up -d --build
```

**Test your running container:**
```bash
# Health check
curl http://localhost:3001/health/ready

# Add a document
curl -X POST http://localhost:3001/doc/add \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Docker makes deployment easy",
    "metadata": {"source": "test"}
  }'

# Search
curl -X POST http://localhost:3001/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deployment",
    "top_k": 5
  }'
```

---

## Phase 4: Understanding Data Persistence

**Goal**: Learn how Docker volumes work and why data persists across container restarts.

### 4.1: What are Docker Volumes?

**Problem**: Containers are ephemeral (temporary). When you delete a container, all its data is lost.

**Solution**: Volumes are storage that exists outside containers.

```
┌─────────────────────┐
│   Docker Host       │
│                     │
│  ┌──────────────┐   │
│  │  Container   │   │
│  │              │   │
│  │  /data ──────┼───┼──> Volume (persists)
│  │              │   │
│  └──────────────┘   │
│                     │
│  Container deleted? │
│  Volume remains! ✓  │
└─────────────────────┘
```

**Types of storage:**
1. **Container Layer**: Temporary, deleted with container
2. **Volume**: Managed by Docker, survives container deletion
3. **Bind Mount**: Maps host directory, good for development

---

### 4.2: Inspect Your Volume

```bash
# List all volumes
docker volume ls

# Expected:
# DRIVER    VOLUME NAME
# local     python_vectorforge-data

# Inspect volume details
docker volume inspect python_vectorforge-data

# Shows:
# - Mount point on host
# - Driver type
# - Creation time
```

---

### 4.3: Test Data Persistence

**Let's verify data survives container restarts:**

```bash
# 1. Add a document to VectorForge
curl -X POST http://localhost:3001/doc/add \
  -H "Content-Type: application/json" \
  -d '{
    "content": "This data should persist across restarts",
    "metadata": {"test": "persistence"}
  }'

# Save the document ID from response
# Example: {"id":"550e8400-e29b-41d4-a716-446655440000","status":"indexed"}

# 2. Verify it exists
curl http://localhost:3001/index/stats

# Note the document_count (should be 1 or more)

# 3. Stop and remove the container
docker compose down

# Container is DELETED but volume remains

# 4. Start a fresh container
docker compose up -d

# 5. Check if data still exists
curl http://localhost:3001/index/stats

# ✅ Document count should be the same!
# ✅ Your data persisted!
```

**What happened:**
1. Documents were written to `/data/chroma` inside container
2. `/data` was mounted from volume `vectorforge-data`
3. Container was deleted but volume remained
4. New container mounted the same volume
5. ChromaDB found existing data in volume

---

### 4.4: Backup Your Data

**Docker volumes can be backed up:**

```bash
# Create a backup of the volume
docker run --rm \
  -v python_vectorforge-data:/data \
  -v $(pwd):/backup \
  ubuntu \
  tar czf /backup/vectorforge-backup.tar.gz /data

# This starts a temporary container that:
# 1. Mounts the vectorforge-data volume as /data
# 2. Mounts current directory as /backup
# 3. Creates a compressed tarball of /data
# 4. Exits and removes itself (--rm)
```

**Restore from backup:**

```bash
# Create new volume
docker volume create vectorforge-data-restored

# Restore backup
docker run --rm \
  -v vectorforge-data-restored:/data \
  -v $(pwd):/backup \
  ubuntu \
  tar xzf /backup/vectorforge-backup.tar.gz -C /
```

---

### 4.5: Development with Bind Mounts

**For development, you might want to mount your code directory:**

**Edit docker-compose.yml, add this under volumes:**
```yaml
    volumes:
      - vectorforge-data:/data
      - ./vectorforge:/app/vectorforge  # Add this for hot-reload
```

**Now code changes reflect immediately (no rebuild needed)!**

---

## Phase 5: Testing and Validation

**Goal**: Verify everything works correctly in Docker.

### 5.1: Run Your Test Suite in Docker

**Create a test Dockerfile:**

**File:** `python/Dockerfile.test`
```dockerfile
FROM vectorforge:latest

# Install dev dependencies
COPY --from=builder /app/.venv /app/.venv
RUN /app/.venv/bin/uv sync --frozen

# Copy tests
COPY tests/ ./tests/
COPY pytest.ini ./

# Run tests
CMD ["pytest", "tests/", "-v"]
```

**Run tests in Docker:**
```bash
# Build test image
docker build -f Dockerfile.test -t vectorforge:test .

# Run tests
docker run --rm vectorforge:test
```

---

### 5.2: Verify All Features Work

**Test document operations:**
```bash
# Add document
DOC_ID=$(curl -s -X POST http://localhost:3001/doc/add \
  -H "Content-Type: application/json" \
  -d '{"content":"Docker test document","metadata":{"test":true}}' \
  | jq -r '.id')

echo "Created document: $DOC_ID"

# Get document
curl http://localhost:3001/doc/$DOC_ID

# Search
curl -X POST http://localhost:3001/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Docker test","top_k":5}'

# Delete document
curl -X DELETE http://localhost:3001/doc/$DOC_ID

# Verify deletion
curl http://localhost:3001/doc/$DOC_ID
# Should return 404
```

---

### 5.3: Test File Upload

```bash
# Create a test text file
echo "This is a test document for VectorForge Docker container." > test.txt

# Upload file
curl -X POST http://localhost:3001/file/upload \
  -F "file=@test.txt"

# Verify chunks were created
curl http://localhost:3001/index/stats

# Clean up
rm test.txt
```

---

### 5.4: Monitor Container Health

```bash
# Check health status
docker inspect vectorforge | jq '.[0].State.Health'

# Expected output:
# {
#   "Status": "healthy",
#   "FailingStreak": 0,
#   "Log": [
#     {
#       "Start": "2024-02-23T15:30:00Z",
#       "End": "2024-02-23T15:30:01Z",
#       "ExitCode": 0,
#       "Output": "..."
#     }
#   ]
# }
```

**Docker will automatically restart unhealthy containers!**

---

### 5.5: Performance Baseline in Docker

**Compare Docker vs local performance:**

```bash
# Install Apache Bench (if not installed)
# sudo apt-get install apache-bench

# Benchmark search endpoint
ab -n 100 -c 10 \
  -p search_payload.json \
  -T application/json \
  http://localhost:3001/search

# Create search_payload.json:
echo '{"query":"test","top_k":5}' > search_payload.json
```

**Expected:** Similar performance to local (1-5% overhead for Docker networking)

---

## Troubleshooting Common Issues

### Issue 0.1: Permission Denied - Cannot Create /data (Local Development)

**Error when running locally:**
```bash
$ uv run vectorforge-api
PermissionError: [Errno 13] Permission denied: '/data'
```

**What happened:**
- The default `CHROMA_DATA_DIR` is `/data/chroma` (for Docker)
- Your user can't create directories in `/` (root filesystem)
- This happens when running locally **without** Docker

**Solution 1: Use the smart default (Recommended - already in guide)**

If you followed the updated Step 1.1, the code automatically detects this:
```python
_default_chroma = "/data/chroma" if os.path.exists("/data") else "./data/chroma"
```

This uses `./data/chroma` locally (no permission error).

**Solution 2: Override with environment variable**
```bash
export CHROMA_DATA_DIR="./data/chroma"
uv run vectorforge-api
```

**Solution 3: Create /data directory with sudo (not recommended)**
```bash
sudo mkdir -p /data
sudo chown $USER:$USER /data
```

---

### Issue 0.2: Cannot Import 'main' from vectorforge.api

**Error when running:**
```bash
$ uv run vectorforge-api
ImportError: cannot import name 'main' from 'vectorforge.api'
```

**What happened:**
- `pyproject.toml` points to wrong location for `main()` function
- Entry point is `vectorforge.api:main` but should be `vectorforge.__main__:main`

**Solution: Fix the entry point**

Edit `python/pyproject.toml`, change line 30:
```toml
# WRONG:
vectorforge-api = "vectorforge.api:main"

# CORRECT:
vectorforge-api = "vectorforge.__main__:main"
```

Then reinstall:
```bash
cd python
uv sync
```

**Why this happens:**
- Python packages define CLI commands in `[project.scripts]`
- Format is `command = "module.path:function_name"`
- The `main()` function lives in `__main__.py`, not `api/__init__.py`

---

### Issue 0.3: Docker Build Fails at `uv sync` - Cannot Find vectorforge Package

**Error during Docker build:**
```bash
$ docker build -t vectorforge:latest .
...
#13 [builder 4/5] RUN uv sync --frozen --no-dev
#13 2.917       ValueError: Unable to determine which files to ship
#13 2.917       inside the wheel using the following heuristics:
#13 2.917       https://hatch.pypa.io/latest/plugins/builder/wheel/#default-file-selection
#13 2.917 
#13 2.917       The most likely cause of this is that there is no directory that matches
#13 2.917       the name of your project (vectorforge).
ERROR: failed to solve: process "/bin/sh -c uv sync --frozen --no-dev" did not complete successfully: exit code: 1
```

**What happened:**
- The Dockerfile copies `pyproject.toml` and `uv.lock` first
- Then runs `uv sync --frozen --no-dev`
- But `pyproject.toml` has `[tool.uv] package = true`, which means `uv` needs to build the local package
- The source code (`vectorforge/` directory) isn't present yet, so the build fails

**Root cause in Dockerfile:**
```dockerfile
# ❌ WRONG ORDER:
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev  # Tries to build vectorforge package, but source is missing!
```

**Solution: Copy source code BEFORE running uv sync**

Edit your `python/Dockerfile`, in the builder stage:

```dockerfile
# ✅ CORRECT ORDER:
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Copy source code (required for package build)
COPY vectorforge/ ./vectorforge/
COPY vectorforge_mcp/ ./vectorforge_mcp/

# Now uv sync can build the package
RUN uv sync --frozen --no-dev
```

**Why this works:**
- `uv sync` with `package = true` installs your local package in editable mode
- It needs the source code to create the `.pth` file and package metadata
- Layer caching still works: dependencies only rebuild if `pyproject.toml` or source changes

**Alternative solution (not recommended):**
You could use `uv sync --no-install-project` followed by a second `uv sync`, but this is more complex.

---

### Issue 0.4: Container Starts But Gets "ModuleNotFoundError: No module named 'vectorforge'"

**Error when running container:**
```bash
$ docker run --rm -p 3001:3001 vectorforge:latest
Traceback (most recent call last):
  File "/app/.venv/bin/vectorforge-api", line 4, in <module>
    from vectorforge.__main__ import main
ModuleNotFoundError: No module named 'vectorforge'
```

**What happened:**
- The Docker image built successfully
- But the container can't find the `vectorforge` module at runtime
- This happens with multi-stage builds when source code is missing in the final stage

**Root cause in Dockerfile:**
```dockerfile
# Builder stage (lines 11-25):
COPY vectorforge/ ./vectorforge/      # ✅ Source code present
RUN uv sync --frozen --no-dev          # ✅ Creates .pth file pointing to /app/vectorforge

# Production stage (lines 27+):
COPY --from=builder /app/.venv /app/.venv           # ✅ Copies .venv (includes .pth file)
COPY --from=builder /root/.cache/huggingface ...   # ✅ Copies model cache
# ❌ MISSING: Never copies the actual vectorforge/ source code!
```

**The Problem:**
- The `.pth` file (in `.venv/lib/python3.11/site-packages/_vectorforge.pth`) says: "look for vectorforge at `/app`"
- But `/app/vectorforge/` doesn't exist in the production stage
- Python can't find the module → `ModuleNotFoundError`

**Solution: Copy source code in production stage too**

Edit your `python/Dockerfile`, in the production stage, add these lines **after** copying the `.venv`:

```dockerfile
FROM base AS production

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /app/.venv /app/.venv

# Copy HuggingFace model cache from builder stage
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# ✅ ADD THESE LINES - Copy source code to final image:
COPY vectorforge/ ./vectorforge/
COPY vectorforge_mcp/ ./vectorforge_mcp/

# Create data directory (where ChromaDB will store data)
RUN mkdir -p /data/chroma

# ... rest of Dockerfile
```

**Why this happens with editable installs:**
- When `uv sync` runs with `package = true`, it installs your package in "editable/development mode"
- This creates a `.pth` file instead of copying the package into `site-packages`
- The `.pth` file is just a pointer to the source code location
- Multi-stage builds require copying the source to both builder AND production stages

**Verification:**
```bash
# Check if source code exists in container:
docker run --rm vectorforge:latest ls -la /app/vectorforge

# Should show:
drwxr-xr-x 1 root root  xxx /app/vectorforge
# ... your Python files
```

**Why we copy source twice:**
- **Builder stage**: Needs it for `uv sync` to build the package
- **Production stage**: Needs it for Python to import the module at runtime
- Both stages need the exact same source code for consistency

---

### Issue 1: Container Exits Immediately

**Check logs:**
```bash
docker logs vectorforge
```

**Common causes:**
- Port already in use (change with `-e API_PORT=3002`)
- Missing environment variables
- Application crash during startup

---

### Issue 2: Cannot Connect to API

**Check if container is running:**
```bash
docker ps | grep vectorforge
```

**Check port mapping:**
```bash
docker port vectorforge
# Should show: 3001/tcp -> 0.0.0.0:3001
```

**Check from inside container:**
```bash
docker exec vectorforge curl http://localhost:3001/health
```

**Common causes:**
- Firewall blocking ports
- Wrong port mapping
- Container not fully started (wait 40s for health check)

---

### Issue 3: Data Not Persisting

**Verify volume exists:**
```bash
docker volume ls | grep vectorforge
```

**Check volume mount:**
```bash
docker inspect vectorforge | jq '.[0].Mounts'
```

**Common causes:**
- Using `--rm` without volume (container data deleted)
- Wrong volume path in docker-compose.yml
- `docker compose down -v` (deletes volumes)

---

### Issue 4: Model Download Slow/Fails

**Pre-download model locally first:**
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Then rebuild image:**
```bash
docker compose build --no-cache
```

---

### Issue 5: Permission Denied Errors

**Solution: Run as non-root user**

Uncomment these lines in Dockerfile:
```dockerfile
RUN useradd -m -u 1000 vectorforge && \
    chown -R vectorforge:vectorforge /app /data
USER vectorforge
```

---

## Next Steps

### Production Deployment Options

1. **Single Server**
   - Use `docker compose` with restart policies
   - Setup reverse proxy (Nginx, Traefik)
   - Configure SSL/TLS certificates

2. **Cloud Platforms**
   - AWS ECS/Fargate
   - Google Cloud Run
   - Azure Container Instances
   - DigitalOcean App Platform

3. **Kubernetes**
   - Create Deployment manifest
   - Setup Persistent Volume Claims
   - Configure Ingress
   - Setup horizontal pod autoscaling

4. **Docker Swarm**
   - Multi-node orchestration
   - Built-in load balancing
   - Rolling updates

---

### Advanced Topics to Explore

1. **Multi-Container Setup**
   - Add Redis for caching
   - Add Postgres for metadata
   - Add Prometheus for metrics

2. **CI/CD Integration**
   - GitHub Actions to build/push images
   - Automated testing in containers
   - Deployment automation

3. **Security Hardening**
   - Image scanning (Trivy, Snyk)
   - Non-root users
   - Read-only filesystems
   - Security contexts

4. **Performance Optimization**
   - Multi-stage builds
   - Layer caching strategies
   - Health check tuning
   - Resource limits

---

## Quick Reference

### Essential Docker Commands

```bash
# Images
docker images                    # List images
docker build -t name:tag .       # Build image
docker rmi image_name            # Remove image
docker pull image_name           # Download image

# Containers
docker ps                        # List running containers
docker ps -a                     # List all containers
docker run [options] image       # Create and start container
docker stop container_name       # Stop container
docker start container_name      # Start stopped container
docker restart container_name    # Restart container
docker rm container_name         # Remove container
docker exec -it container bash   # Open shell in container

# Logs and Inspection
docker logs container_name       # View logs
docker logs -f container_name    # Follow logs
docker inspect container_name    # Detailed info
docker stats                     # Resource usage

# Volumes
docker volume ls                 # List volumes
docker volume inspect volume     # Volume details
docker volume rm volume          # Remove volume
docker volume prune              # Remove unused volumes

# Docker Compose
docker compose up                # Start services
docker compose up -d             # Start detached
docker compose down              # Stop and remove
docker compose logs              # View logs
docker compose ps                # List services
docker compose build             # Rebuild images
```

---

### VectorForge Docker Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHROMA_DATA_DIR` | `/data/chroma` | ChromaDB storage location |
| `HF_HOME` | `/root/.cache/huggingface` | Model cache directory |
| `API_PORT` | `3001` | API server port |
| `API_HOST` | `0.0.0.0` | API bind address |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |

---

## Congratulations! 🎉

You've successfully:
- ✅ Learned Docker fundamentals
- ✅ Containerized VectorForge
- ✅ Understood volumes and persistence
- ✅ Created production-ready Docker setup
- ✅ Mastered troubleshooting techniques

**Your VectorForge API is now:**
- Portable (runs anywhere Docker runs)
- Reproducible (same environment every time)
- Scalable (easy to deploy multiple instances)
- Production-ready (health checks, persistence, logging)

---

## Resources

- [Official Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Best Practices for Writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Docker Security Best Practices](https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html)

---

**Need Help?**
- Review the Troubleshooting section
- Check container logs: `docker logs vectorforge`
- Inspect container: `docker inspect vectorforge`
- Test from inside container: `docker exec -it vectorforge bash`
