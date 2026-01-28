"""Entry point when running vectorforge as a module"""

import uvicorn

from vectorforge.config import VFConfig


def main() -> None:
    """Entry point for the VectorForge API server"""
    VFConfig.validate()
    uvicorn.run(
        "vectorforge.api:app", 
        host=VFConfig.API_HOST, 
        port=VFConfig.API_PORT
    )


if __name__ == "__main__":
    main()
