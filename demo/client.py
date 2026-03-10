"""Shared HTTP session, request helpers, and user-input utilities for the demo."""

import json
from io import BufferedReader
from typing import Any

import requests

BASE_URL = "http://localhost:3001"

_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})


def get(path: str, params: dict[str, Any] | None = None) -> requests.Response:
    """Send a GET request to the API.

    Args:
      path: URL path relative to BASE_URL (e.g. "/collections").
      params: Optional query parameters passed to requests as a dict.

    Returns:
      The raw requests.Response object.
    """
    return _session.get(f"{BASE_URL}{path}", params=params)


def post(
    path: str, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
) -> requests.Response:
    """Send a POST request with a JSON body to the API.

    Args:
      path: URL path relative to BASE_URL.
      body: Optional dict serialised as JSON.
      params: Optional query parameters passed to requests as a dict.

    Returns:
      The raw requests.Response object.
    """
    return _session.post(f"{BASE_URL}{path}", json=body, params=params)


def put(
    path: str, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
) -> requests.Response:
    """Send a PUT request with a JSON body to the API.

    Args:
      path: URL path relative to BASE_URL.
      body: Optional dict serialised as JSON.
      params: Optional query parameters passed to requests as a dict.

    Returns:
      The raw requests.Response object.
    """
    return _session.put(f"{BASE_URL}{path}", json=body, params=params)


def delete(
    path: str, body: dict[str, Any] | None = None, params: dict[str, Any] | None = None
) -> requests.Response:
    """Send a DELETE request to the API, with an optional JSON body.

    Args:
      path: URL path relative to BASE_URL.
      body: Optional dict serialised as JSON (used for batch deletes).
      params: Optional query parameters passed to requests as a dict.

    Returns:
      The raw requests.Response object.
    """
    return _session.delete(f"{BASE_URL}{path}", json=body, params=params)


def post_file(
    path: str,
    file_field: tuple[str, BufferedReader, str],
    data: dict[str, Any] | None = None,
) -> requests.Response:
    """Send a multipart/form-data POST request for file uploads.

    Omits the session-level JSON Content-Type header so that requests can set
    the correct multipart boundary automatically.

    Args:
      path: URL path relative to BASE_URL.
      file_field: Tuple of (filename, file_object, mime_type) for the file field.
      data: Optional additional form fields to include alongside the file.

    Returns:
      The raw requests.Response object.
    """
    return _session.post(
        f"{BASE_URL}{path}",
        files={"file": file_field},
        data=data or {},
        headers={},
    )


def print_response(resp: requests.Response) -> None:
    """Print the HTTP status code and pretty-printed response body.

    Args:
      resp: The response returned by one of the request helpers.
    """
    status_label = _status_label(resp.status_code)
    print(f"\n  {status_label}  {resp.status_code} {resp.reason}")

    try:
        body = resp.json()
        print(json.dumps(body, indent=2))
    except ValueError:
        print(resp.text or "(empty body)")

    print()


def _status_label(code: int) -> str:
    """Return a short bracket label describing the HTTP status class.

    Args:
      code: HTTP status code.

    Returns:
      One of "[OK]", "[REDIRECT]", "[CLIENT ERROR]", or "[SERVER ERROR]".
    """
    if code < 300:
        return "[OK]"
    if code < 400:
        return "[REDIRECT]"
    if code < 500:
        return "[CLIENT ERROR]"

    return "[SERVER ERROR]"


def prompt(label: str, default: str | None = None) -> str:
    """Prompt the user for a required string value.

    Args:
      label: Text displayed before the input cursor.
      default: Value returned when the user presses Enter without typing.
        If None, the user must provide input (no default shown).

    Returns:
      The entered string, or default if the input is blank and default is set.
    """
    if default is not None:
        raw = input(f"  {label} [{default}]: ").strip()
        return raw if raw else default

    raw = input(f"  {label}: ").strip()
    return raw


def prompt_optional(label: str) -> str | None:
    """Prompt for an optional string value, returning None if the user skips it.

    Args:
      label: Text displayed before the input cursor.

    Returns:
      The entered string, or None if the input is blank.
    """
    raw = input(f"  {label} (optional, press Enter to skip): ").strip()
    return raw if raw else None


def prompt_int(label: str, default: int | None = None) -> int | None:
    """Prompt for an integer value, with an optional default.

    Args:
      label: Text displayed before the input cursor.
      default: Value returned when the user presses Enter without typing.
        If None and the user skips, None is returned.

    Returns:
      The parsed integer, the default, or None if skipped with no default.
    """
    suffix = (
        f" [{default}]" if default is not None else " (optional, press Enter to skip)"
    )
    raw = input(f"  {label}{suffix}: ").strip()

    if not raw:
        return default

    try:
        return int(raw)
    except ValueError:
        print(
            f"  (invalid integer, using {'default' if default is not None else 'None'})"
        )
        return default


def prompt_bool(label: str, default: bool = False) -> bool:
    """Prompt for a yes/no confirmation.

    Args:
      label: Text displayed before the input cursor.
      default: Value returned when the user presses Enter without typing.

    Returns:
      True if the user enters y/yes/true/1, False otherwise.
    """
    default_str = "Y/n" if default else "y/N"
    raw = input(f"  {label} [{default_str}]: ").strip().lower()

    if not raw:
        return default

    return raw in ("y", "yes", "true", "1")


def prompt_collection() -> str:
    """Prompt for a collection name, defaulting to the built-in 'vectorforge' collection.

    Returns:
      The collection name entered by the user.
    """
    return prompt("Collection name", default="vectorforge")


def prompt_json(label: str) -> dict[str, Any] | None:
    """Prompt for a JSON object, returning None if blank or unparseable.

    Args:
      label: Text displayed before the input cursor.

    Returns:
      Parsed dict, or None if the input is blank or not a valid JSON object.
    """
    raw = input(f"  {label} (JSON object, optional, press Enter to skip): ").strip()

    if not raw:
        return None

    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            print("  (expected a JSON object, ignoring)")
            return None

        return parsed
    except json.JSONDecodeError as exc:
        print(f"  (invalid JSON — {exc}, ignoring)")
        return None
