"""Thin HTTP client for communicating with the VectorForge REST API.

All functions call raise_for_status() so callers receive requests.HTTPError
on any non-2xx response.
"""

from typing import Any

import requests

from vectorforge_mcp.config import MCPConfig


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
    response = requests.get(_url(path), params=params)
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
    response = requests.post(
        _url(path), json=json, params=params, files=files, data=data
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
    response = requests.put(_url(path), json=json, params=params)
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
    response = requests.delete(_url(path), json=json, params=params)
    response.raise_for_status()
    return response.json()
