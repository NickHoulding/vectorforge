"""FastMCP instance for VectorForge MCP Server."""
from fastmcp import FastMCP

from vectorforge import __version__


mcp = FastMCP(
    name="VectorForge MCP Server",
    version=__version__
)
