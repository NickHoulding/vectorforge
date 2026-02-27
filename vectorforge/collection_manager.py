"""Collection Manager for multi-collection support in VectorForge.

Manages multiple ChromaDB collections and their associated VectorEngine instances.
Provides collection lifecycle operations (create, list, delete) and engine caching.
"""

import logging
import os
import re
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import chromadb
import chromadb.errors
from sentence_transformers import SentenceTransformer

from vectorforge.config import VFGConfig
from vectorforge.models.collections import CollectionInfo
from vectorforge.models.index import HNSWConfig
from vectorforge.vector_engine import VectorEngine

logger = logging.getLogger(__name__)


class CollectionManager:
    """Manages multiple ChromaDB collections and engine instances.

    Lightweight manager that creates VectorEngine instances on-demand and caches
    them for reuse. Handles collection lifecycle (CRUD) and enforces limits.

    Attributes:
        chroma_client: ChromaDB PersistentClient for database operations
        model: Shared SentenceTransformer model (memory optimization)

    Example:
        >>> manager = CollectionManager()
        >>> engine = manager.get_engine("my_collection")
        >>> collections = manager.list_collections()
    """

    META_DESCRIPTION_KEY = "vf:description"
    META_CREATED_AT_KEY = "vf:created_at"
    META_PREFIX = "vf:meta:"

    def __init__(self) -> None:
        """Initialize the CollectionManager with ChromaDB client and shared model."""
        chroma_path: str = VFGConfig.CHROMA_PERSIST_DIR

        if not os.path.isabs(chroma_path):
            chroma_path = os.path.abspath(chroma_path)

        os.makedirs(chroma_path, exist_ok=True)

        self.chroma_path: str = chroma_path
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        logger.info(f"Loading shared model: {VFGConfig.MODEL_NAME}")
        self.model: SentenceTransformer = SentenceTransformer(VFGConfig.MODEL_NAME)
        self._engine_cache: dict[str, VectorEngine] = {}
        self._cache_lock = Lock()

        self._ensure_default_collection()
        logger.info(
            f"CollectionManager initialized (max_collections={VFGConfig.MAX_COLLECTIONS})"
        )

    def _ensure_default_collection(self) -> None:
        """Ensure the default collection exists on startup."""
        try:
            self.chroma_client.get_collection(name=VFGConfig.DEFAULT_COLLECTION_NAME)
            logger.info(
                f"Default collection '{VFGConfig.DEFAULT_COLLECTION_NAME}' exists"
            )
        except chromadb.errors.NotFoundError:
            logger.info(
                f"Creating default collection: {VFGConfig.DEFAULT_COLLECTION_NAME}"
            )
            self.create_collection(
                name=VFGConfig.DEFAULT_COLLECTION_NAME,
                hnsw_config={},
                description="Default VectorForge collection",
                metadata={},
            )

    def validate_collection_name(self, name: str) -> None:
        """Validate collection name format.

        Args:
            name: Collection name to validate

        Raises:
            ValueError: If name is invalid (format, length, characters, uniqueness)
        """
        if not name:
            raise ValueError("Collection name cannot be empty")

        if len(name) < VFGConfig.MIN_COLLECTION_NAME_LENGTH:
            raise ValueError(
                f"Collection name must be at least {VFGConfig.MIN_COLLECTION_NAME_LENGTH} character(s)"
            )

        if len(name) > VFGConfig.MAX_COLLECTION_NAME_LENGTH:
            raise ValueError(
                f"Collection name must be at most {VFGConfig.MAX_COLLECTION_NAME_LENGTH} characters"
            )

        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(
                "Collection name must contain only alphanumeric characters, underscores, and hyphens"
            )

        if self.collection_exists(name=name):
            raise ValueError(f"Collection with name: {name} already exists.")

    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists.

        Args:
            name: Collection name to check

        Returns:
            True if collection exists, False otherwise
        """
        try:
            self.chroma_client.get_collection(name=name)
            return True
        except Exception:
            return False

    def get_engine(self, collection_name: str | None = None) -> VectorEngine:
        """Get or create VectorEngine for a collection.

        Retrieves engine from cache if available, otherwise creates new one.
        Engines are cached for performance (avoid repeated model loading).

        Args:
            collection_name: Name of collection (uses default if None)

        Returns:
            VectorEngine instance for the collection

        Raises:
            ValueError: If collection doesn't exist
        """
        if collection_name is None:
            collection_name = VFGConfig.DEFAULT_COLLECTION_NAME

        with self._cache_lock:
            if collection_name in self._engine_cache:
                logger.debug(f"Using cached engine for collection: {collection_name}")
                return self._engine_cache[collection_name]

            if not self.collection_exists(collection_name):
                raise ValueError(f"Collection '{collection_name}' does not exist")

            collection = self.chroma_client.get_collection(name=collection_name)

            logger.info(f"Creating new engine for collection: {collection_name}")
            engine = VectorEngine(
                collection=collection,
                model=self.model,
                chroma_client=self.chroma_client,
            )

            if len(self._engine_cache) >= VFGConfig.COLLECTION_CACHE_SIZE:
                oldest_key = next(iter(self._engine_cache))
                logger.debug(f"Cache full, evicting oldest: {oldest_key}")
                del self._engine_cache[oldest_key]

            self._engine_cache[collection_name] = engine

        return engine

    def list_collections(self) -> list[CollectionInfo]:
        """List all collections with their metadata.

        Returns:
            List of CollectionInfo objects with details about each collection

        Example:
            >>> collections = manager.list_collections()
            >>> for col in collections:
            ...     print(f"{col.name}: {col.document_count} docs")
        """
        collections: list[CollectionInfo] = []

        for collection in self.chroma_client.list_collections():
            info = self._get_collection_info(collection)
            collections.append(info)

        collections.sort(key=lambda collection: collection.collection_name)

        return collections

    def _get_collection_info(self, collection: chromadb.Collection) -> CollectionInfo:
        """Extract CollectionInfo from a ChromaDB collection.

        Args:
            collection: ChromaDB Collection object

        Returns:
            CollectionInfo with all metadata
        """
        config = collection.configuration
        hnsw_config_dict: dict[str, Any] = dict(config.get("hnsw") or {})

        hnsw_config = HNSWConfig(
            space=str(hnsw_config_dict.get("space", "cosine")),
            ef_construction=int(hnsw_config_dict.get("ef_construction", 100)),
            ef_search=int(hnsw_config_dict.get("ef_search", 100)),
            max_neighbors=int(hnsw_config_dict.get("max_neighbors", 16)),
            resize_factor=float(hnsw_config_dict.get("resize_factor", 1.2)),
            sync_threshold=int(hnsw_config_dict.get("sync_threshold", 1000)),
        )

        metadata_dict = collection.metadata or {}
        description = metadata_dict.get(self.META_DESCRIPTION_KEY, None)
        created_at = metadata_dict.get(
            self.META_CREATED_AT_KEY, datetime.now(timezone.utc).isoformat()
        )

        custom_metadata = {}
        for key, value in metadata_dict.items():
            if key.startswith(self.META_PREFIX):
                clean_key = key[len(self.META_PREFIX) :]
                custom_metadata[clean_key] = value

        return CollectionInfo(
            collection_name=collection.name,
            id=str(collection.id),
            document_count=collection.count(),
            created_at=created_at,
            description=description,
            hnsw_config=hnsw_config,
            metadata=custom_metadata,
        )

    def get_collection_info(self, name: str) -> CollectionInfo:
        """Get detailed information about a collection.

        Args:
            name: Collection name

        Returns:
            CollectionInfo with all metadata

        Raises:
            ValueError: If collection doesn't exist
        """
        if not self.collection_exists(name):
            raise ValueError(f"Collection '{name}' does not exist")

        collection = self.chroma_client.get_collection(name=name)
        return self._get_collection_info(collection)

    def create_collection(
        self,
        name: str,
        hnsw_config: dict[str, Any],
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CollectionInfo:
        """Create a new collection with HNSW config and metadata.

        Args:
            name: Collection name (validated)
            hnsw_config: HNSW parameters (uses defaults for missing keys)
            description: Optional human-readable description
            metadata: Optional custom metadata (up to 20 key-value pairs)

        Returns:
            CollectionInfo for the newly created collection

        Raises:
            ValueError: If name is invalid or collection already exists
            RuntimeError: If max collections limit reached
        """
        if self.collection_exists(name):
            raise ValueError(f"Collection '{name}' already exists")

        if self.get_collection_count() >= VFGConfig.MAX_COLLECTIONS:
            raise RuntimeError(
                f"Maximum collections limit ({VFGConfig.MAX_COLLECTIONS}) reached"
            )

        hnsw_metadata = {
            "hnsw:space": hnsw_config.get("space", "cosine"),
            "hnsw:construction_ef": hnsw_config.get("ef_construction", 100),
            "hnsw:search_ef": hnsw_config.get("ef_search", 100),
            "hnsw:M": hnsw_config.get("max_neighbors", 16),
            "hnsw:resize_factor": hnsw_config.get("resize_factor", 1.2),
            "hnsw:sync_threshold": hnsw_config.get("sync_threshold", 1000),
        }

        if description:
            hnsw_metadata[self.META_DESCRIPTION_KEY] = description

        hnsw_metadata[self.META_CREATED_AT_KEY] = datetime.now(timezone.utc).isoformat()

        if metadata:
            if len(metadata) > VFGConfig.MAX_METADATA_PAIRS:
                raise ValueError(
                    f"Metadata cannot exceed {VFGConfig.MAX_METADATA_PAIRS} key-value pairs"
                )
            for key, value in metadata.items():
                hnsw_metadata[f"{self.META_PREFIX}{key}"] = str(value)

        logger.info(f"Creating collection: {name}")
        collection = self.chroma_client.create_collection(
            name=name, metadata=hnsw_metadata
        )

        logger.info(f"Collection '{name}' created successfully")

        return self._get_collection_info(collection)

    def delete_collection(self, name: str) -> None:
        """Delete a collection and remove from cache.

        Args:
            name: Collection name to delete

        Raises:
            ValueError: If collection doesn't exist
        """
        if not self.collection_exists(name):
            raise ValueError(f"Collection '{name}' does not exist")

        logger.info(f"Deleting collection: {name}")

        with self._cache_lock:
            if name in self._engine_cache:
                logger.debug(f"Removing {name} from engine cache")
                del self._engine_cache[name]

        self.chroma_client.delete_collection(name=name)
        logger.info(f"Collection '{name}' deleted successfully")

    def get_collection_count(self) -> int:
        """Get the total number of collections.

        Returns:
            Number of collections in the database
        """
        return len(self.chroma_client.list_collections())
