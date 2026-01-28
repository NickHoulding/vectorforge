class APIConfig:
    """Configuration for VectorForge API server.
    
    Contains all configurable constants for the FastAPI server including
    host, port, and other API-specific settings.
    """
    
    # =============================================================================
    # API Configuration
    # =============================================================================
    
    API_PORT: int = 3001
    """Default port for the FastAPI server."""
    
    API_HOST: str = "0.0.0.0"
    """Default host binding for the FastAPI server."""
    
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
