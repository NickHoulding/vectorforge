"""Allows running the demo as a module: 'uv run demo' from the project root"""

from demo.main import ensure_container_ready, run_repl


def main() -> None:
    """Ensure the container is ready, then start the interactive demo."""
    ensure_container_ready()
    run_repl()


if __name__ == "__main__":
    main()
