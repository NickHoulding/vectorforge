# TODO Items
- [] Review and finalize the load test script and results (stashed)

## Naming / simplification refactor candidates (surveyed 2026-09-19)

Worth fixing soon (real correctness risk):
- [x] `vectorforge_mcp/tools/files.py`'s `list_files` reads `data.get("files", [])` for its log line, but the REST response field is actually `filenames` (`FileListResponse.filenames`) — log always reports 0 files; caller payload is unaffected. Fixed: now reads `data.get("filenames", [])`.
- [x] `LOG_LEVEL` name collision across config classes: `VFGConfig.LOG_LEVEL` and `MCPConfig.LOG_LEVEL` both derive from `VF_LOG_LEVEL`, but `vectorforge/api/config.py`'s `APIConfig.LOG_LEVEL` reads from an unprefixed `LOG_LEVEL` env var instead — same attribute name, three different env vars behind it. Turned out to be dead code (never read anywhere, not validated). Fixed: deleted `APIConfig.LOG_LEVEL`; corrected `Dockerfile`, `docker-compose.yml`, and `README.md`'s env var table to use `VF_LOG_LEVEL` (the variable that actually works — already correct in `.env.example`).

Found while fixing the above (not yet fixed, same category):
- [x] `README.md`'s env var table also documented `MAX_COLLECTIONS` and `COLLECTION_CACHE_SIZE` without the `VF_` prefix — same dead-env-var bug pattern as `LOG_LEVEL` had. Fixed: table now shows `VF_MAX_COLLECTIONS`/`VF_COLLECTION_CACHE_SIZE`, matching code and `.env.example`.
- [x] `vectorforge_mcp/README.md`'s "MCPConfig Settings" table (~line 510-522) was stale: most rows said Env Var "(none)" but actually have working `VF_`-prefixed env vars, and it listed a nonexistent `LOG_FORMAT` setting while missing 5 real ones. Fixed: table now lists all 11 actual `MCPConfig` fields with correct env vars and defaults; `LOG_FORMAT` dropped.

Same spirit as the add/delete merge + Batch* rename:
- [x] `doc_ids` vs `ids` inconsistency: `DocumentsResponse`/`DocumentIdsInput` use `ids`, but `FileUploadResponse`/`FileDeleteResponse` used `doc_ids` for the same concept. Fixed: renamed to `ids` throughout (models, API, `VectorEngine.delete_file()`, tests, README) — MCP layer needed no changes since it just passes the response through.
- [] `doc`/`docs` vs `document`/`documents` naming split: REST functions and `VectorEngine` methods say `get_doc`, `add_docs`, `delete_docs`; the MCP tools calling the same endpoints say `get_document`, `add_documents`, `delete_documents`.
- [] `GET /collections/{name}/files/list` is the only list-shaped endpoint with a redundant `/list` suffix (compare `GET /collections`, `GET /documents`).

Lower priority / nice-to-have:
- [] Stats/metrics naming spread across four names (`get_collection_stats`, `get_index_stats`, `get_collection_metrics`, `get_metrics`) for what's really two concepts, inconsistent across REST/engine/MCP layers.
- [] `search` (REST function name) vs `search_documents` (MCP tool name) — the one REST/MCP pair that doesn't share a name.
- [] Stale docstring in `vectorforge/models/metadata.py` references a nonexistent `add_doc()` function.
- [] Leftover "batch add"/"batch delete" wording in comments/docstrings (`config.py`, `vector_engine.py`, `api/documents.py`) that survived the `Batch*` model rename.
