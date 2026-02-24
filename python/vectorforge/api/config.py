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
        assert isinstance(cls.API_PORT, int)
        assert 1 <= cls.API_PORT <= 65535

        assert isinstance(cls.API_HOST, str)
        assert len(cls.API_HOST) > 0
        assert not cls.API_HOST.isspace()
