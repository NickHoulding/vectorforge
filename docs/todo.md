## Multi-Collection Refactor: Cleanup

- [x] **#1 Hardcoded temp name in HNSW migration** — `vector_engine.py:688` always uses `vectorforge_temp_...` regardless of which collection is being migrated; use the actual collection name instead
- [x] **#2 Wrong ChromaDB key for `sync_threshold`** — `collection_manager.py:293` maps `sync_threshold` to `"hnsw:batch_size"` (incorrect); correct key is `"hnsw:sync_threshold"`; same bug in `vector_engine.py:680–686`
- [x] **#4 Redundant double existence check in `create_collection()`** — N/A: `validate_collection_name()` does check existence, so the second check at `collection_manager.py:278` is dead code but harmless; left as-is since removing it is low value
- [x] **#5 TOCTOU race condition in `get_engine()`** — `collection_manager.py:152` calls `collection_exists()` outside the cache lock; two threads can both create engines for the same collection
- [x] **#6 Private attribute accessed from API layer** — `api/index.py:119` reads `engine._migration_in_progress` directly; should be a public property on `VectorEngine`
- [x] **#7 `require_collection` decorator has no error protection** — `api/decorators.py`: decorator sits above `@handle_api_errors` so a ChromaDB error inside `collection_exists()` propagates uncaught
- [x] **#8 Duplicate config constants** — `config.py:108` `CHROMA_COLLECTION_NAME` and `config.py:121` `DEFAULT_COLLECTION_NAME` are identical; remove the former
- [x] **#9 Dead config constants** — `config.py:25–35` `DEFAULT_DATA_DIR`, `METADATA_FILENAME`, `EMBEDDINGS_FILENAME` are never used; remove them and their assertions in `validate()`
- [x] **#10 Dead branch in `VectorEngine.__init__`** — `vector_engine.py:154` checks `if chroma_client:` but the param is typed `ClientAPI` (non-optional); the `else` branch is unreachable
- [x] **#11 Pydantic `ConfigDict` defined as inner class** — `models/collections.py` uses `class ConfigDict` as a nested class instead of `model_config = ConfigDict(...)`; `json_schema_extra` examples are silently ignored
- [x] **#12 LRU cache is actually FIFO** — `collection_manager.py:163` evicts oldest-inserted key (`next(iter(...))`) not least-recently-used; rename or implement true LRU
- [x] **#13 Inconsistent `confirm` parameter type** — `api/collections.py` delete uses `Optional[bool]`; `api/index.py` HNSW update uses `Optional[str]`; pick one pattern
- [x] **#14 Inconsistent MCP `collection_name` parameter** — `vectorforge_mcp/tools/collections.py:119` `delete_collection` uses param name `name`; all other MCP tools use `collection_name`
- [x] **#15 Bare `except Exception` at startup** — `collection_manager.py:73` swallows all exceptions in `_ensure_default_collection`, masking ChromaDB connectivity failures

## Todo
- [ ] **Deploy on DigitalOcean**
- [ ] **Pivot metric persistence for lifetime tracking instead of current session-scoped setup**
- [ ] **Work on updating benchmark suite to benchmark new ChromaDB integrated version**

### Future Enhancements
- [ ] **Custom distance functions** - Expose ChromaDB's support for different distance metrics beyond cosine similarity
- [ ] **Hybrid search exploration** - Investigate ChromaDB's built-in hybrid search capabilities (vector + keyword)
