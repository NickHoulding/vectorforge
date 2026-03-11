"""Launcher script that ensures the VectorForge Docker container is running before starting the MCP server.

Intended to be used as the Claude Desktop command so the API is always available
when the MCP server starts.
"""

import subprocess
import sys
import time
from pathlib import Path

import requests

from vectorforge_mcp.config import MCPConfig

# Docker Compose service name (must match docker-compose.yml)
_SERVICE_NAME = "vectorforge"

# How long to wait for the API to become healthy after docker compose up (seconds)
_HEALTH_TIMEOUT = 30

# Interval between health check polls (seconds)
_HEALTH_POLL_INTERVAL = 2


def _find_project_root() -> Path:
    """Walk up from this file's location to find the directory containing docker-compose.yml.

    Returns:
      Path to the project root directory.

    Raises:
      RuntimeError: If docker-compose.yml is not found in any ancestor directory.
    """
    current = Path(__file__).resolve().parent

    while True:
        if (current / "docker-compose.yml").exists():
            return current

        if current.parent == current:
            raise RuntimeError(
                "Could not find docker-compose.yml. "
                "Ensure the MCP server is installed from the VectorForge project root."
            )

        current = current.parent


def _docker_daemon_running() -> bool:
    """Check whether the Docker daemon is reachable.

    Returns:
      True if 'docker info' exits successfully, False otherwise.
    """
    result = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return result.returncode == 0


def _container_running(project_root: Path) -> bool:
    """Check whether the VectorForge service is currently running.

    Args:
      project_root: Path to the directory containing docker-compose.yml.

    Returns:
      True if the service is listed as running, False otherwise.
    """
    result = subprocess.run(
        ["docker", "compose", "ps", "--services", "--filter", "status=running"],
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    running_services = result.stdout.splitlines()
    return _SERVICE_NAME in running_services


def _start_container(project_root: Path) -> bool:
    """Start the VectorForge container and wait for the API to become healthy.

    Args:
      project_root: Path to the directory containing docker-compose.yml.

    Returns:
      True if the API became healthy within the timeout, False otherwise.
    """
    print("Starting VectorForge container...", file=sys.stderr)

    result = subprocess.run(
        ["docker", "compose", "up", "-d"],
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if result.returncode != 0:
        print("docker compose up failed.", file=sys.stderr)
        return False

    health_url = f"{MCPConfig.VECTORFORGE_API_BASE_URL}/health"
    deadline = time.monotonic() + _HEALTH_TIMEOUT

    print(f"Waiting for VectorForge API at {health_url}...", file=sys.stderr)

    while time.monotonic() < deadline:
        try:
            response = requests.get(health_url, timeout=2)

            if response.ok:
                print("VectorForge API is ready.", file=sys.stderr)
                return True
        except requests.RequestException:
            pass

        time.sleep(_HEALTH_POLL_INTERVAL)

    print(
        f"VectorForge API did not become healthy within {_HEALTH_TIMEOUT}s.",
        file=sys.stderr,
    )

    return False


def main() -> None:
    """Ensure the VectorForge container is running, then start the MCP server."""
    try:
        project_root = _find_project_root()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if not _docker_daemon_running():
        print(
            "Error: Docker daemon is not running. Please start Docker and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    if _container_running(project_root):
        print("VectorForge container is already running.", file=sys.stderr)
    else:
        if not _start_container(project_root):
            sys.exit(1)

    result = subprocess.run(["vectorforge-mcp"])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
