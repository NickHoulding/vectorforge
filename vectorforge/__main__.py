"""Entry point when running vectorforge as a module"""

import logging

from dotenv import load_dotenv

load_dotenv()

from vectorforge.logging import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

from vectorforge.api.config import APIConfig
from vectorforge.config import VFGConfig


def main() -> None:
    """Entry point for the VectorForge API server"""
    import uvicorn

    VFGConfig.validate()
    APIConfig.validate()

    logger.info(
        "Starting VectorForge API server on %s:%d",
        APIConfig.API_HOST,
        APIConfig.API_PORT,
    )

    uvicorn.run("vectorforge.api:app", host=APIConfig.API_HOST, port=APIConfig.API_PORT)


if __name__ == "__main__":
    main()
