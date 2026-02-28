import logging


class MCPConfig:
    """Configuration for VectorForge MCP Server.

    Contains configurable settings for the Model Context Protocol server
    including server metadata and logging configuration.
    """

    # =============================================================================
    # Server Metadata
    # =============================================================================

    SERVER_NAME: str = "VectorForge MCP Server"
    """Name of the MCP server shown to clients."""

    SERVER_DESCRIPTION: str = (
        "Model Context Protocol server for VectorForge vector database"
    )
    """Description of the MCP server."""

    # =============================================================================
    # Logging Configuration
    # =============================================================================

    LOG_LEVEL: int = logging.INFO
    """Logging level for MCP server."""

    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    """Format string for log messages."""

    # =============================================================================
    # Configuration Class Validator
    # =============================================================================

    @classmethod
    def validate(cls) -> None:
        """Validate configuration values."""
        if not isinstance(cls.SERVER_NAME, str):
            raise ValueError("SERVER_NAME must be a string")
        if len(cls.SERVER_NAME) == 0:
            raise ValueError("SERVER_NAME cannot be empty")

        if not isinstance(cls.SERVER_DESCRIPTION, str):
            raise ValueError("SERVER_DESCRIPTION must be a string")
        if len(cls.SERVER_DESCRIPTION) == 0:
            raise ValueError("SERVER_DESCRIPTION cannot be empty")

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
