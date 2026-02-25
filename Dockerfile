FROM python:3.11-slim AS base

ENV PYTHONBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

ENV HF_HOME=/app/.cache/huggingface

COPY pyproject.toml uv.lock ./
COPY vectorforge/ ./vectorforge/
COPY vectorforge_mcp/ ./vectorforge_mcp/

RUN uv sync --frozen --no-dev

RUN mkdir -p /app/.cache/huggingface && \
    uv run python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

FROM base AS production

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.cache/huggingface /home/vectorforge/.cache/huggingface

COPY vectorforge/ ./vectorforge
COPY vectorforge_mcp/ ./vectorforge_mcp/

RUN mkdir -p /data/chroma

ENV CHROMA_DATA_DIR=/data/chroma \
    HF_HOME=/home/vectorforge/.cache/huggingface \
    API_PORT=3001 \
    API_HOST=0.0.0.0 \
    LOG_LEVEL=INFO \
    PATH="/app/.venv/bin:$PATH"

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 CMD curl -f http://localhost:3001/health/live || exit 1

RUN useradd -m -u 1000 vectorforge && \
    chown -R vectorforge:vectorforge /app /data /home/vectorforge
USER vectorforge

CMD ["vectorforge-api"]
