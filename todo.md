# TODO Items
- [] Review and finalize the load test script and results (stashed)

## Naming / simplification refactor candidates (surveyed 2026-09-19)

Worth fixing soon (real correctness risk):
- [x] `vectorforge_mcp/tools/files.py`'s `list_files` reads `data.get("files", [])` for its log line, but the REST response field is actually `filenames` (`FileListResponse.filenames`) — log always reports 0 files; caller payload is unaffected. Fixed: now reads `data.get("filenames", [])`.
- [] `LOG_LEVEL` name collision across config classes: `VFGConfig.LOG_LEVEL` and `MCPConfig.LOG_LEVEL` both derive from `VF_LOG_LEVEL`, but `vectorforge/api/config.py`'s `APIConfig.LOG_LEVEL` reads from an unprefixed `LOG_LEVEL` env var instead — same attribute name, three different env vars behind it.

Same spirit as the add/delete merge + Batch* rename:
- [] `doc_ids` vs `ids` inconsistency: `DocumentsResponse`/`DocumentIdsInput` use `ids`, but `FileUploadResponse`/`FileDeleteResponse` use `doc_ids` for the same concept.
- [] `doc`/`docs` vs `document`/`documents` naming split: REST functions and `VectorEngine` methods say `get_doc`, `add_docs`, `delete_docs`; the MCP tools calling the same endpoints say `get_document`, `add_documents`, `delete_documents`.
- [] `GET /collections/{name}/files/list` is the only list-shaped endpoint with a redundant `/list` suffix (compare `GET /collections`, `GET /documents`).

Lower priority / nice-to-have:
- [] Stats/metrics naming spread across four names (`get_collection_stats`, `get_index_stats`, `get_collection_metrics`, `get_metrics`) for what's really two concepts, inconsistent across REST/engine/MCP layers.
- [] `search` (REST function name) vs `search_documents` (MCP tool name) — the one REST/MCP pair that doesn't share a name.
- [] Stale docstring in `vectorforge/models/metadata.py` references a nonexistent `add_doc()` function.
- [] Leftover "batch add"/"batch delete" wording in comments/docstrings (`config.py`, `vector_engine.py`, `api/documents.py`) that survived the `Batch*` model rename.
