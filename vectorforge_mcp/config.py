"""Configuration for the VectorForge MCP server."""

import logging
import os

logger = logging.getLogger(__name__)


class MCPConfig:
    """Static configuration for the VectorForge MCP Server.

    Contains all tuneable settings for server metadata, API connection,
    collection defaults, and logging.
    """

    # =============================================================================
    # Server Metadata
    # =============================================================================

    SERVER_NAME: str = os.environ.get("VF_SERVER_NAME", "VectorForge MCP Server")
    """Display name for the MCP server."""

    SERVER_DESCRIPTION: str = os.environ.get(
        "VF_SERVER_DESCRIPTION",
        "Model Context Protocol server for VectorForge vector database",
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
    # Collection Defaults
    # =============================================================================

    DEFAULT_COLLECTION_NAME: str = os.environ.get(
        "VF_DEFAULT_COLLECTION_NAME", "vectorforge"
    )
    """Default collection used when no collection_name argument is provided to a tool."""

    DEFAULT_TOP_K: int = int(os.environ.get("VF_DEFAULT_TOP_K", "10"))
    """Default number of search results returned by search_documents."""

    # =============================================================================
    # Logging
    # =============================================================================

    _LOG_LEVEL_STR: str = os.environ.get("VF_LOG_LEVEL", "INFO").upper()
    LOG_LEVEL: int = getattr(logging, _LOG_LEVEL_STR, logging.INFO)
    """Logging verbosity level for the MCP server process."""

    LOG_FILE: str = os.environ.get("VF_LOG_FILE", ".logs/vectorforge_mcp.log")
    """Path to log file (relative or absolute)."""

    LOG_JSON_CONSOLE: bool = (
        os.environ.get("VF_LOG_JSON_CONSOLE", "false").lower() == "true"
    )
    """Enable JSON console output (useful for production log aggregation)."""

    LOG_MAX_TEXT_LEN: int = int(os.environ.get("VF_LOG_MAX_TEXT_LEN", "100"))
    """Maximum text length for sanitization in logs (default: 100 chars)."""

    LOG_MAX_BYTES: int = int(os.environ.get("VF_LOG_MAX_BYTES", "10485760"))
    """Maximum size per log file before rotation in bytes (default: 10MB)."""

    LOG_BACKUP_COUNT: int = int(os.environ.get("VF_LOG_BACKUP_COUNT", "5"))
    """Number of backup log files to keep (default: 5, total 50MB)."""

    # =============================================================================
    # Validation
    # =============================================================================

    @classmethod
    def validate(cls) -> None:
        """Validate all configuration values at startup.

        Raises:
            ValueError: If any setting has an invalid type or value.
        """
        logger.info("Starting VectorForge MCP configuration validation")

        logger.debug("Validating SERVER_NAME: %s", cls.SERVER_NAME)
        if not isinstance(cls.SERVER_NAME, str):
            logger.error("Validation failed: SERVER_NAME must be a string")
            raise ValueError("SERVER_NAME must be a string")
        if len(cls.SERVER_NAME) == 0:
            logger.error("Validation failed: SERVER_NAME cannot be empty")
            raise ValueError("SERVER_NAME cannot be empty")

        logger.debug("Validating SERVER_DESCRIPTION: %s", cls.SERVER_DESCRIPTION[:50])
        if not isinstance(cls.SERVER_DESCRIPTION, str):
            logger.error("Validation failed: SERVER_DESCRIPTION must be a string")
            raise ValueError("SERVER_DESCRIPTION must be a string")
        if len(cls.SERVER_DESCRIPTION) == 0:
            logger.error("Validation failed: SERVER_DESCRIPTION cannot be empty")
            raise ValueError("SERVER_DESCRIPTION cannot be empty")

        logger.debug(
            "Validating VECTORFORGE_API_BASE_URL: %s", cls.VECTORFORGE_API_BASE_URL
        )
        if not isinstance(cls.VECTORFORGE_API_BASE_URL, str):
            logger.error("Validation failed: VECTORFORGE_API_BASE_URL must be a string")
            raise ValueError("VECTORFORGE_API_BASE_URL must be a string")
        if len(cls.VECTORFORGE_API_BASE_URL) == 0:
            logger.error("Validation failed: VECTORFORGE_API_BASE_URL cannot be empty")
            raise ValueError("VECTORFORGE_API_BASE_URL cannot be empty")

        logger.debug(
            "Validating DEFAULT_COLLECTION_NAME: %s", cls.DEFAULT_COLLECTION_NAME
        )
        if not isinstance(cls.DEFAULT_COLLECTION_NAME, str):
            logger.error("Validation failed: DEFAULT_COLLECTION_NAME must be a string")
            raise ValueError("DEFAULT_COLLECTION_NAME must be a string")
        if len(cls.DEFAULT_COLLECTION_NAME) == 0:
            logger.error("Validation failed: DEFAULT_COLLECTION_NAME cannot be empty")
            raise ValueError("DEFAULT_COLLECTION_NAME cannot be empty")

        logger.debug("Validating DEFAULT_TOP_K: %d", cls.DEFAULT_TOP_K)
        if not isinstance(cls.DEFAULT_TOP_K, int):
            logger.error("Validation failed: DEFAULT_TOP_K must be an int")
            raise ValueError("DEFAULT_TOP_K must be an int")
        if cls.DEFAULT_TOP_K <= 0:
            logger.error(
                "Validation failed: DEFAULT_TOP_K=%d must be > 0", cls.DEFAULT_TOP_K
            )
            raise ValueError("DEFAULT_TOP_K must be > 0")

        logger.debug("Validating LOG_LEVEL: %s", logging.getLevelName(cls.LOG_LEVEL))
        if not isinstance(cls.LOG_LEVEL, int):
            logger.error("Validation failed: LOG_LEVEL must be an int")
            raise ValueError("LOG_LEVEL must be an int")
        if cls.LOG_LEVEL not in (
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ):
            logger.error(
                "Validation failed: LOG_LEVEL=%d is not a valid logging level",
                cls.LOG_LEVEL,
            )
            raise ValueError(
                "LOG_LEVEL must be a valid logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
            )

        logger.debug("Validating LOG_FILE: %s", cls.LOG_FILE)
        if not isinstance(cls.LOG_FILE, str):
            logger.error("Validation failed: LOG_FILE must be a string")
            raise ValueError("LOG_FILE must be a string")
        if len(cls.LOG_FILE) == 0:
            logger.error("Validation failed: LOG_FILE cannot be empty")
            raise ValueError("LOG_FILE cannot be empty")

        logger.debug("Validating LOG_JSON_CONSOLE: %s", cls.LOG_JSON_CONSOLE)
        if not isinstance(cls.LOG_JSON_CONSOLE, bool):
            logger.error("Validation failed: LOG_JSON_CONSOLE must be a bool")
            raise ValueError("LOG_JSON_CONSOLE must be a bool")

        logger.debug("Validating LOG_MAX_TEXT_LEN: %d", cls.LOG_MAX_TEXT_LEN)
        if not isinstance(cls.LOG_MAX_TEXT_LEN, int):
            logger.error("Validation failed: LOG_MAX_TEXT_LEN must be an int")
            raise ValueError("LOG_MAX_TEXT_LEN must be an int")
        if cls.LOG_MAX_TEXT_LEN <= 0:
            logger.error(
                "Validation failed: LOG_MAX_TEXT_LEN=%d must be > 0",
                cls.LOG_MAX_TEXT_LEN,
            )
            raise ValueError("LOG_MAX_TEXT_LEN must be > 0")

        logger.debug("Validating LOG_MAX_BYTES: %d", cls.LOG_MAX_BYTES)
        if not isinstance(cls.LOG_MAX_BYTES, int):
            logger.error("Validation failed: LOG_MAX_BYTES must be an int")
            raise ValueError("LOG_MAX_BYTES must be an int")
        if cls.LOG_MAX_BYTES <= 0:
            logger.error(
                "Validation failed: LOG_MAX_BYTES=%d must be > 0", cls.LOG_MAX_BYTES
            )
            raise ValueError("LOG_MAX_BYTES must be > 0")

        logger.debug("Validating LOG_BACKUP_COUNT: %d", cls.LOG_BACKUP_COUNT)
        if not isinstance(cls.LOG_BACKUP_COUNT, int):
            logger.error("Validation failed: LOG_BACKUP_COUNT must be an int")
            raise ValueError("LOG_BACKUP_COUNT must be an int")
        if cls.LOG_BACKUP_COUNT < 0:
            logger.error(
                "Validation failed: LOG_BACKUP_COUNT=%d must be >= 0",
                cls.LOG_BACKUP_COUNT,
            )
            raise ValueError("LOG_BACKUP_COUNT must be >= 0")

        logger.info("Configuration validation completed successfully")
