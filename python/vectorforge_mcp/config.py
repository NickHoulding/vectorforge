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
    
    SERVER_DESCRIPTION: str = "Model Context Protocol server for VectorForge vector database"
    """Description of the MCP server."""
    
    # =============================================================================
    # VectorForge API Connection
    # =============================================================================
    
    VECTORFORGE_API_BASE_URL: str = "http://localhost:3001"
    """Base URL for VectorForge API connections (documentation only)."""
    
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
        assert isinstance(cls.SERVER_NAME, str)
        assert len(cls.SERVER_NAME) > 0
        
        assert isinstance(cls.SERVER_DESCRIPTION, str)
        assert len(cls.SERVER_DESCRIPTION) > 0
        
        assert isinstance(cls.VECTORFORGE_API_BASE_URL, str)
        assert cls.VECTORFORGE_API_BASE_URL.startswith("http")
        
        assert isinstance(cls.LOG_LEVEL, int)
        assert cls.LOG_LEVEL in (
            logging.DEBUG, 
            logging.INFO, 
            logging.WARNING, 
            logging.ERROR, 
            logging.CRITICAL
        )
        
        assert isinstance(cls.LOG_FORMAT, str)
        assert len(cls.LOG_FORMAT) > 0
