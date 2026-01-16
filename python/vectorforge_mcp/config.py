import logging


class MCPConfig:
    """Configuration for VectorForge MCP Server.
    
    Contains all configurable settings for the Model Context Protocol server
    including server metadata, API connection settings, timeout values,
    and logging configuration.
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
    
    VECTORFORGE_API_HOST: str = "localhost"
    """Host where VectorForge API is running."""
    
    VECTORFORGE_API_PORT: int = 3001
    """Port where VectorForge API is running."""
    
    VECTORFORGE_API_BASE_URL: str = f"http://{VECTORFORGE_API_HOST}:{VECTORFORGE_API_PORT}"
    """Base URL for VectorForge API connections."""
    
    # =============================================================================
    # Connection Settings
    # =============================================================================
    
    API_TIMEOUT: float = 30.0
    """Timeout in seconds for API requests."""
    
    API_MAX_RETRIES: int = 3
    """Maximum number of retry attempts for failed API calls."""
    
    API_RETRY_DELAY: float = 1.0
    """Delay in seconds between retry attempts."""
    
    CONNECTION_POOL_SIZE: int = 10
    """Maximum number of concurrent connections to VectorForge API."""
    
    # =============================================================================
    # File Upload Settings
    # =============================================================================
    
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB
    """Maximum file upload size in bytes."""
    
    UPLOAD_TIMEOUT: float = 60.0
    """Timeout in seconds for file upload operations."""
    
    ALLOWED_UPLOAD_EXTENSIONS: tuple[str, ...] = (".pdf", ".txt")
    """Tuple of allowed file extensions for upload."""
    
    # =============================================================================
    # Tool Execution Settings
    # =============================================================================
    
    TOOL_EXECUTION_TIMEOUT: float = 45.0
    """Maximum execution time for individual MCP tool calls."""
    
    ENABLE_ASYNC_TOOLS: bool = True
    """Whether to enable async execution for MCP tools."""
    
    # =============================================================================
    # Logging Configuration
    # =============================================================================
    
    LOG_LEVEL: int = logging.INFO
    """Logging level for MCP server."""
    
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    """Format string for log messages."""
    
    LOG_TO_FILE: bool = False
    """Whether to write logs to a file."""
    
    LOG_FILE_PATH: str = "./logs/mcp_server.log"
    """Path to log file if LOG_TO_FILE is enabled."""
    
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    """Maximum size of log file before rotation."""
    
    LOG_BACKUP_COUNT: int = 5
    """Number of backup log files to keep."""
    
    # =============================================================================
    # Error Handling
    # =============================================================================
    
    INCLUDE_ERROR_DETAILS: bool = True
    """Whether to include detailed error information in responses."""
    
    INCLUDE_STACK_TRACES: bool = False
    """Whether to include stack traces in error responses (dev mode)."""
    
    SANITIZE_ERROR_MESSAGES: bool = True
    """Whether to sanitize error messages to prevent information leakage."""
    
    # =============================================================================
    # Response Settings
    # =============================================================================
    
    MAX_RESPONSE_SIZE: int = 1024 * 1024  # 1 MB
    """Maximum size of tool response in bytes."""
    
    TRUNCATE_LARGE_RESPONSES: bool = True
    """Whether to truncate responses exceeding MAX_RESPONSE_SIZE."""
    
    # =============================================================================
    # Health Check Settings
    # =============================================================================
    
    HEALTH_CHECK_INTERVAL: float = 30.0
    """Interval in seconds between health checks to VectorForge API."""
    
    HEALTH_CHECK_ENABLED: bool = True
    """Whether to enable periodic health checks."""
    
    HEALTH_CHECK_TIMEOUT: float = 5.0
    """Timeout for health check requests."""
    
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
        
        assert isinstance(cls.VECTORFORGE_API_HOST, str)
        assert len(cls.VECTORFORGE_API_HOST) > 0
        
        assert isinstance(cls.VECTORFORGE_API_PORT, int)
        assert 1 <= cls.VECTORFORGE_API_PORT <= 65535
        
        assert isinstance(cls.VECTORFORGE_API_BASE_URL, str)
        assert cls.VECTORFORGE_API_BASE_URL.startswith("http")
        
        assert isinstance(cls.API_TIMEOUT, (int, float))
        assert cls.API_TIMEOUT > 0
        
        assert isinstance(cls.API_MAX_RETRIES, int)
        assert cls.API_MAX_RETRIES >= 0
        
        assert isinstance(cls.API_RETRY_DELAY, (int, float))
        assert cls.API_RETRY_DELAY >= 0
        
        assert isinstance(cls.CONNECTION_POOL_SIZE, int)
        assert cls.CONNECTION_POOL_SIZE > 0
        
        assert isinstance(cls.MAX_UPLOAD_SIZE, int)
        assert cls.MAX_UPLOAD_SIZE > 0
        
        assert isinstance(cls.UPLOAD_TIMEOUT, (int, float))
        assert cls.UPLOAD_TIMEOUT > 0
        
        assert isinstance(cls.ALLOWED_UPLOAD_EXTENSIONS, tuple)
        assert len(cls.ALLOWED_UPLOAD_EXTENSIONS) > 0
        assert all(
            isinstance(ext, str) 
            and ext.startswith(".") 
            and len(ext) > 1
            for ext in cls.ALLOWED_UPLOAD_EXTENSIONS
        )
        
        assert isinstance(cls.TOOL_EXECUTION_TIMEOUT, (int, float))
        assert cls.TOOL_EXECUTION_TIMEOUT > 0
        
        assert isinstance(cls.ENABLE_ASYNC_TOOLS, bool)
        
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
        
        assert isinstance(cls.LOG_TO_FILE, bool)
        
        assert isinstance(cls.LOG_FILE_PATH, str)
        assert len(cls.LOG_FILE_PATH) > 0
        
        assert isinstance(cls.LOG_MAX_BYTES, int)
        assert cls.LOG_MAX_BYTES > 0
        
        assert isinstance(cls.LOG_BACKUP_COUNT, int)
        assert cls.LOG_BACKUP_COUNT >= 0
        
        assert isinstance(cls.INCLUDE_ERROR_DETAILS, bool)
        assert isinstance(cls.INCLUDE_STACK_TRACES, bool)
        assert isinstance(cls.SANITIZE_ERROR_MESSAGES, bool)
        
        assert isinstance(cls.MAX_RESPONSE_SIZE, int)
        assert cls.MAX_RESPONSE_SIZE > 0
        
        assert isinstance(cls.TRUNCATE_LARGE_RESPONSES, bool)
        
        assert isinstance(cls.HEALTH_CHECK_INTERVAL, (int, float))
        assert cls.HEALTH_CHECK_INTERVAL > 0
        
        assert isinstance(cls.HEALTH_CHECK_ENABLED, bool)
        
        assert isinstance(cls.HEALTH_CHECK_TIMEOUT, (int, float))
        assert cls.HEALTH_CHECK_TIMEOUT > 0
        
        # Cross-validation
        assert cls.UPLOAD_TIMEOUT <= cls.TOOL_EXECUTION_TIMEOUT
        assert cls.HEALTH_CHECK_TIMEOUT < cls.API_TIMEOUT
