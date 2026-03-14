# Module Docstring Reference

Every Python file must begin with a module-level docstring as its very first statement, before any imports.

## Template

```python
"""Short description of what this module does.

Contains <primary classes / functions / groups> for <purpose>.
"""
```

The second line is optional. Omit it if the single-sentence description is sufficient.

## Examples

**Simple utility module (one line is enough):**
```python
"""Text chunking and PDF extraction utilities for document pre-processing."""
```

**Module with several related classes:**
```python
"""Pydantic request and response models for the search API.

Covers SearchQuery, SearchResult, and SearchResponse.
"""
```

**Router/endpoint module:**
```python
"""FastAPI router for document CRUD endpoints.

Handles /collections/{name}/documents routes: add, get, delete.
"""
```

**Configuration module:**
```python
"""Application configuration loaded from environment variables.

Exposes a single VFGConfig dataclass with all tuneable settings.
"""
```
