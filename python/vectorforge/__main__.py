"""Entry point when running vectorforge as a module"""

import uvicorn

from vectorforge.api.config import APIConfig


def main() -> None:
    """Entry point for the VectorForge API server"""
    APIConfig.validate()
    uvicorn.run("vectorforge.api:app", host=APIConfig.API_HOST, port=APIConfig.API_PORT)


if __name__ == "__main__":
    main()
