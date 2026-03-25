"""MCP server entry point - registers all tool modules and runs the server."""

import logging

from vectorforge_mcp.config import MCPConfig
from vectorforge_mcp.logging import configure_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the VectorForge MCP server console script."""
    configure_logging(
        log_level=MCPConfig.LOG_LEVEL,
        log_file=MCPConfig.LOG_FILE,
        json_console=MCPConfig.LOG_JSON_CONSOLE,
    )

    logger.info("Starting VectorForge MCP server")
    MCPConfig.validate()

    from vectorforge_mcp.instance import mcp
    from vectorforge_mcp.tools import (
        collections,
        documents,
        files,
        index,
        search,
        system,
    )

    logger.info("All tools registered, starting MCP server")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
