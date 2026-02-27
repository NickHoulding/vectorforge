import os


class APIConfig:
    """Configuration for VectorForge API server.

    Contains all configurable constants for the FastAPI server including
    host, port, and other API-specific settings.
    """

    # =============================================================================
    # API Configuration
    # =============================================================================

    API_PORT: int = int(os.getenv("API_PORT", "3001"))
    """Default port for the FastAPI server."""

    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    """Default host binding for the FastAPI server."""

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    """Logging level (DEBUG, INFO, WARNING, ERROR). Configurable via LOG_LEVEL env var."""

    # =============================================================================
    # Configuration Class Validator
    # =============================================================================

    @classmethod
    def validate(cls) -> None:
        """Validate configuration values."""
        if not isinstance(cls.API_PORT, int):
            raise ValueError("API_PORT must be an int")
        if not (1 <= cls.API_PORT <= 65535):
            raise ValueError("API_PORT must be between 1 and 65535")

        if not isinstance(cls.API_HOST, str):
            raise ValueError("API_HOST must be a string")
        if len(cls.API_HOST) == 0:
            raise ValueError("API_HOST cannot be empty")
        if cls.API_HOST.isspace():
            raise ValueError("API_HOST cannot be whitespace")
