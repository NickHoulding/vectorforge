"""Entry point when running vectorforge as a module"""

import uvicorn

from vectorforge.api.config import APIConfig
from vectorforge.config import VFGConfig


def main() -> None:
    """Entry point for the VectorForge API server"""
    VFGConfig.validate()
    APIConfig.validate()
    uvicorn.run("vectorforge.api:app", host=APIConfig.API_HOST, port=APIConfig.API_PORT)


if __name__ == "__main__":
    main()
