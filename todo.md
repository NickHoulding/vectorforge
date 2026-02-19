## ChromaDB Integration TODOs (Post-Migration)
- [ ] **Explicit backup/restore API** - (Convert existing save/load/build endpoints) Implement explicit save/load endpoints that backup/restore ChromaDB data directory (currently auto-persists, endpoints are no-ops)
- [ ] **Update metrics** - Remove deprecated metrics
- [ ] **Add ChromaDB-specific metrics** - Integrate ChromaDB collection stats, index health, and telemetry into `/metrics` endpoint

### Future Enhancements
- [ ] **Collection management API** - Add endpoints for creating/deleting multiple collections (multi-index support)
- [ ] **Custom distance functions** - Expose ChromaDB's support for different distance metrics beyond cosine similarity
- [ ] **Hybrid search exploration** - Investigate ChromaDB's built-in hybrid search capabilities (vector + keyword)
