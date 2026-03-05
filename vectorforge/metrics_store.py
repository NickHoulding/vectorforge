"""SQLite-backed persistent store for lifetime VectorEngine metrics.

Provides per-collection metric rows that survive engine restarts. All counters
and timestamps are stored in a single ``metrics`` table keyed by collection
name, co-located with the ChromaDB data directory.
"""

import sqlite3
from contextlib import contextmanager
from typing import Any, Generator

# Columns that hold integer counters (support atomic INCREMENT).
_COUNTER_COLUMNS: frozenset[str] = frozenset(
    {
        "total_queries",
        "docs_added",
        "docs_deleted",
        "chunks_created",
        "files_uploaded",
        "total_documents_peak",
    }
)

# Columns that hold floating-point accumulators (support atomic INCREMENT).
_FLOAT_COLUMNS: frozenset[str] = frozenset({"total_query_time_ms"})

# Columns that hold signed byte-counts (may decrease; use direct SET).
_SIZE_COLUMNS: frozenset[str] = frozenset({"total_doc_size_bytes"})

# All persisted numeric / text columns (excludes PK and session-only fields).
_ALL_COLUMNS: tuple[str, ...] = (
    "total_queries",
    "docs_added",
    "docs_deleted",
    "chunks_created",
    "files_uploaded",
    "total_query_time_ms",
    "total_doc_size_bytes",
    "total_documents_peak",
    "created_at",
    "last_query_at",
    "last_doc_added_at",
    "last_file_uploaded_at",
)

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS metrics (
    collection_name       TEXT PRIMARY KEY,
    total_queries         INTEGER NOT NULL DEFAULT 0,
    docs_added            INTEGER NOT NULL DEFAULT 0,
    docs_deleted          INTEGER NOT NULL DEFAULT 0,
    chunks_created        INTEGER NOT NULL DEFAULT 0,
    files_uploaded        INTEGER NOT NULL DEFAULT 0,
    total_query_time_ms   REAL    NOT NULL DEFAULT 0.0,
    total_doc_size_bytes  INTEGER NOT NULL DEFAULT 0,
    total_documents_peak  INTEGER NOT NULL DEFAULT 0,
    created_at            TEXT    NOT NULL,
    last_query_at         TEXT,
    last_doc_added_at     TEXT,
    last_file_uploaded_at TEXT
)
"""


class MetricsStore:
    """SQLite-backed persistence layer for lifetime VectorEngine metrics.

    Opens (or creates) a ``metrics.db`` file inside ``db_path`` and exposes
    methods to load, save, and atomically increment per-collection counters.
    Each collection gets exactly one row, keyed by ``collection_name``.

    Thread safety is handled by SQLite's serialised WAL mode; concurrent
    reads are fine and writes are serialised automatically.

    Attributes:
        db_path: Filesystem path to the ``metrics.db`` SQLite database file.
    """

    def __init__(self, db_path: str) -> None:
        """Open or create the metrics database.

        Args:
            db_path: Full path to the SQLite file (e.g.
                ``/data/chroma/metrics.db``). The parent directory must
                already exist.
        """
        self.db_path = db_path
        self._init_db()

    # =========================================================================
    # Internal helpers
    # =========================================================================

    def _init_db(self) -> None:
        """Create the metrics table if it does not already exist."""
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield an open, auto-committing SQLite connection.

        Yields:
            An open ``sqlite3.Connection`` in WAL mode. Commits on clean
            exit and rolls back on exception.
        """
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # =========================================================================
    # Public API
    # =========================================================================

    def load(self, collection_name: str) -> dict[str, Any] | None:
        """Load persisted metrics for a collection.

        Args:
            collection_name: Name of the collection to look up.

        Returns:
            Dictionary of column values if the row exists, ``None`` otherwise.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM metrics WHERE collection_name = ?",
                (collection_name,),
            ).fetchone()

        if row is None:
            return None

        return dict(row)

    def insert(self, collection_name: str, created_at: str) -> None:
        """Insert a zero-valued metrics row for a new collection.

        This is a no-op if a row already exists (INSERT OR IGNORE).

        Args:
            collection_name: Name of the collection.
            created_at: ISO UTC timestamp to record as the lifetime
                ``created_at`` value for this collection.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO metrics (collection_name, created_at)
                VALUES (?, ?)
                """,
                (collection_name, created_at),
            )

    def save(self, collection_name: str, data: dict[str, Any]) -> None:
        """Overwrite persisted metric fields for a collection.

        Only the keys present in ``data`` are written; unknown keys are
        silently ignored. The ``collection_name`` (PK) is never overwritten.

        Args:
            collection_name: Name of the collection row to update.
            data: Mapping of column-name to new value.
        """
        valid = {k: v for k, v in data.items() if k in _ALL_COLUMNS}

        if not valid:
            return

        set_clause = ", ".join(f"{col} = ?" for col in valid)
        values = list(valid.values()) + [collection_name]

        with self._connect() as conn:
            conn.execute(
                f"UPDATE metrics SET {set_clause} WHERE collection_name = ?",
                values,
            )

    def increment(self, collection_name: str, field: str, delta: int | float) -> None:
        """Atomically increment a numeric counter column.

        Uses a single ``UPDATE … SET col = col + delta`` statement so that
        concurrent increments cannot produce lost-update anomalies.

        Args:
            collection_name: Name of the collection row to update.
            field: Column name to increment. Must be in
                ``_COUNTER_COLUMNS`` or ``_FLOAT_COLUMNS``.
            delta: Value to add (positive) or subtract (negative).

        Raises:
            ValueError: If ``field`` is not an incrementable column.
        """
        if field not in _COUNTER_COLUMNS and field not in _FLOAT_COLUMNS:
            raise ValueError(
                f"Field '{field}' is not an incrementable metrics column. "
                f"Use save() for direct assignment."
            )

        with self._connect() as conn:
            conn.execute(
                f"UPDATE metrics SET {field} = {field} + ? WHERE collection_name = ?",
                (delta, collection_name),
            )
