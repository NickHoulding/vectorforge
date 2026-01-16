from .instance import mcp
from .tools import documents, files, index, search, system


def main() -> None:
    """Entry point for the VectorForge MCP server console script."""
    mcp.run()


if __name__ == "__main__":
    main()
