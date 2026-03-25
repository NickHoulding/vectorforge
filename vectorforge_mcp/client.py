"""Thin HTTP client for communicating with the VectorForge REST API.

All functions call raise_for_status() so callers receive requests.HTTPError
on any non-2xx response.
"""

import logging
import time
from typing import Any

import requests

from vectorforge_mcp.config import MCPConfig

logger = logging.getLogger(__name__)


def _url(path: str) -> str:
    """Build a full API URL from a path segment.

    Args:
      path: URL path, e.g. '/collections/vectorforge/documents'.

    Returns:
      Full URL string with the configured base URL prepended.
    """
    return f"{MCPConfig.VECTORFORGE_API_BASE_URL}{path}"


def get(path: str, params: dict[str, Any] | None = None) -> Any:
    """Send a GET request and return the parsed JSON response.

    Args:
      path: API path relative to the base URL.
      params: Optional query parameters.

    Returns:
      Parsed JSON response body.
    """
    url = _url(path)
    logger.debug("GET request: path=%s, params=%s", path, params)

    start_time = time.time()
    response = requests.get(url, params=params)
    elapsed = time.time() - start_time

    logger.info(
        "GET %s completed: status=%d, duration=%.2fs",
        path,
        response.status_code,
        elapsed,
    )
    response.raise_for_status()
    return response.json()


def post(
    path: str,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> Any:
    """Send a POST request and return the parsed JSON response.

    Args:
      path: API path relative to the base URL.
      json: JSON request body.
      params: Optional query parameters.
      files: Optional multipart file payload.
      data: Optional form data payload.

    Returns:
      Parsed JSON response body.
    """
    url = _url(path)
    logger.debug(
        "POST request: path=%s, has_json=%s, has_files=%s, has_data=%s",
        path,
        json is not None,
        files is not None,
        data is not None,
    )

    start_time = time.time()
    response = requests.post(url, json=json, params=params, files=files, data=data)
    elapsed = time.time() - start_time

    logger.info(
        "POST %s completed: status=%d, duration=%.2fs",
        path,
        response.status_code,
        elapsed,
    )
    response.raise_for_status()
    return response.json()


def put(
    path: str,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    """Send a PUT request and return the parsed JSON response.

    Args:
      path: API path relative to the base URL.
      json: JSON request body.
      params: Optional query parameters.

    Returns:
      Parsed JSON response body.
    """
    url = _url(path)
    logger.debug("PUT request: path=%s, has_json=%s", path, json is not None)

    start_time = time.time()
    response = requests.put(url, json=json, params=params)
    elapsed = time.time() - start_time

    logger.info(
        "PUT %s completed: status=%d, duration=%.2fs",
        path,
        response.status_code,
        elapsed,
    )
    response.raise_for_status()
    return response.json()


def delete(
    path: str,
    json: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    """Send a DELETE request and return the parsed JSON response.

    Args:
      path: API path relative to the base URL.
      json: Optional JSON request body.
      params: Optional query parameters.

    Returns:
      Parsed JSON response body.
    """
    url = _url(path)
    logger.debug("DELETE request: path=%s, has_json=%s", path, json is not None)

    start_time = time.time()
    response = requests.delete(url, json=json, params=params)
    elapsed = time.time() - start_time

    logger.info(
        "DELETE %s completed: status=%d, duration=%.2fs",
        path,
        response.status_code,
        elapsed,
    )
    response.raise_for_status()
    return response.json()
