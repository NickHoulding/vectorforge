"""Configuration for the VectorForge MCP server."""

import logging
import os


class MCPConfig:
    """Static configuration for the VectorForge MCP Server.

    Contains all tuneable settings for server metadata, API connection,
    collection defaults, and logging.
    """

    # =============================================================================
    # Server Metadata
    # =============================================================================

    SERVER_NAME: str = "VectorForge MCP Server"
    """Display name for the MCP server."""

    SERVER_DESCRIPTION: str = (
        "Model Context Protocol server for VectorForge vector database"
    )
    """Short description of the MCP server reported to MCP clients."""

    # =============================================================================
    # VectorForge API Connection
    # =============================================================================

    VECTORFORGE_API_BASE_URL: str = os.environ.get(
        "VECTORFORGE_API_BASE_URL", "http://localhost:3001"
    )
    """Base URL of the VectorForge REST API. Override via VECTORFORGE_API_BASE_URL env var."""

    # =============================================================================
    # MCP Server Binding
    # =============================================================================

    MCP_HOST: str = os.environ.get("MCP_HOST", "0.0.0.0")
    """Network interface the MCP SSE server binds to. Override via MCP_HOST env var."""

    MCP_PORT: int = int(os.environ.get("MCP_PORT", "3002"))
    """TCP port the MCP SSE server listens on. Override via MCP_PORT env var."""

    # =============================================================================
    # Collection Defaults
    # =============================================================================

    DEFAULT_COLLECTION_NAME: str = "vectorforge"
    """Default collection used when no collection_name argument is provided to a tool."""

    DEFAULT_TOP_K: int = 10
    """Default number of search results returned by search_documents."""

    # =============================================================================
    # Logging
    # =============================================================================

    LOG_LEVEL: int = logging.INFO
    """Logging verbosity level for the MCP server process."""

    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    """Log record format string passed to logging.basicConfig."""

    # =============================================================================
    # Validation
    # =============================================================================

    @classmethod
    def validate(cls) -> None:
        """Validate all configuration values at startup.

        Raises:
            ValueError: If any setting has an invalid type or value.
        """
        if not isinstance(cls.SERVER_NAME, str):
            raise ValueError("SERVER_NAME must be a string")
        if len(cls.SERVER_NAME) == 0:
            raise ValueError("SERVER_NAME cannot be empty")

        if not isinstance(cls.SERVER_DESCRIPTION, str):
            raise ValueError("SERVER_DESCRIPTION must be a string")
        if len(cls.SERVER_DESCRIPTION) == 0:
            raise ValueError("SERVER_DESCRIPTION cannot be empty")

        if not isinstance(cls.VECTORFORGE_API_BASE_URL, str):
            raise ValueError("VECTORFORGE_API_BASE_URL must be a string")
        if len(cls.VECTORFORGE_API_BASE_URL) == 0:
            raise ValueError("VECTORFORGE_API_BASE_URL cannot be empty")

        if not isinstance(cls.DEFAULT_COLLECTION_NAME, str):
            raise ValueError("DEFAULT_COLLECTION_NAME must be a string")
        if len(cls.DEFAULT_COLLECTION_NAME) == 0:
            raise ValueError("DEFAULT_COLLECTION_NAME cannot be empty")

        if not isinstance(cls.DEFAULT_TOP_K, int):
            raise ValueError("DEFAULT_TOP_K must be an int")
        if cls.DEFAULT_TOP_K <= 0:
            raise ValueError("DEFAULT_TOP_K must be > 0")

        if not isinstance(cls.LOG_LEVEL, int):
            raise ValueError("LOG_LEVEL must be an int")
        if cls.LOG_LEVEL not in (
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ):
            raise ValueError(
                "LOG_LEVEL must be a valid logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
            )

        if not isinstance(cls.LOG_FORMAT, str):
            raise ValueError("LOG_FORMAT must be a string")
        if len(cls.LOG_FORMAT) == 0:
            raise ValueError("LOG_FORMAT cannot be empty")

        if not isinstance(cls.MCP_HOST, str):
            raise ValueError("MCP_HOST must be a string")
        if len(cls.MCP_HOST) == 0:
            raise ValueError("MCP_HOST cannot be empty")

        if not isinstance(cls.MCP_PORT, int):
            raise ValueError("MCP_PORT must be an int")
        if not (1 <= cls.MCP_PORT <= 65535):
            raise ValueError("MCP_PORT must be between 1 and 65535")
