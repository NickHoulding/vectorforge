"""FastMCP instance for VectorForge MCP Server."""

from fastmcp import FastMCP

from vectorforge_mcp.config import MCPConfig

mcp: FastMCP = FastMCP(name=MCPConfig.SERVER_NAME)
