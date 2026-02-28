import logging

from vectorforge_mcp.config import MCPConfig
from vectorforge_mcp.instance import mcp
from vectorforge_mcp.tools import collections, documents, files, index, search, system


def main() -> None:
    """Entry point for the VectorForge MCP server console script."""
    MCPConfig.validate()
    logging.basicConfig(level=MCPConfig.LOG_LEVEL, format=MCPConfig.LOG_FORMAT)
    mcp.run()


if __name__ == "__main__":
    main()
