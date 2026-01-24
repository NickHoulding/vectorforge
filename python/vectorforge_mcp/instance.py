"""FastMCP instance for VectorForge MCP Server."""
from fastmcp import FastMCP

from vectorforge import __version__
from vectorforge_mcp.config import MCPConfig


mcp: FastMCP = FastMCP(
    name=MCPConfig.SERVER_NAME,
    version=__version__
)
