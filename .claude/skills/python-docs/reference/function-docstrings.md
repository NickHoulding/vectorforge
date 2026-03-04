# Function & Method Docstring Reference

All functions and methods must have a Google-style docstring placed immediately after the `def` line.

## Template

```python
def function_name(param1, param2):
    """One or two concise sentences describing what the function does.

    Args:
      param1: Short description of param1.
      param2: Short description of param2.

    Returns:
      Short description of the return value.

    Raises:
      SomeError: When condition that triggers this error occurs.

    Example:
      Short example showing how to call this function and what it does.
    """
    ...
```

Omit any section that does not apply. Never leave a section header with no content.

## Section rules

| Section | Include when |
|---------|-------------|
| `Args:` | Function has one or more parameters |
| `Returns:` | Function returns anything other than `None` |
| `Raises:` | Function can raise an exception the caller should know about |
| `Example:` | Usage is genuinely non-obvious (complex setup, surprising behaviour) |

## Examples

**No parameters, no return value:**
```python
def reset():
    """Clear all cached state and reset counters to zero."""
    ...
```

**Parameters and return value:**
```python
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks of a fixed character size.

    Args:
      text: The input text to split.
      chunk_size: Maximum number of characters per chunk.
      overlap: Number of characters to repeat between consecutive chunks.

    Returns:
      List of text chunks in order.
    """
    ...
```

**With Raises:**
```python
def get_doc(doc_id: str) -> dict:
    """Retrieve a document by its unique ID.

    Args:
      doc_id: UUID of the document to fetch.

    Returns:
      Dictionary with 'content' and 'metadata' keys.

    Raises:
      ValueError: If doc_id is empty or not a valid UUID format.
    """
    ...
```

**With Example (non-obvious usage):**
```python
def update_hnsw_config(new_config: dict) -> dict:
    """Update HNSW index configuration via a zero-downtime collection migration.

    Args:
      new_config: Partial HNSW settings to apply. Unspecified keys keep defaults.

    Returns:
      Migration result with status, statistics, and new config.

    Raises:
      RuntimeError: If a migration is already in progress.

    Example:
      result = engine.update_hnsw_config({"ef_search": 150, "space": "l2"})
      print(result["migration"]["documents_migrated"])
    """
    ...
```
