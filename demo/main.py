"""Demo entry point: verifies the Docker container is healthy, then starts the REPL.

Run with: 'uv run demo' from the project root.
"""

import subprocess
import sys
import time

import requests

from demo.features import FEATURES

CONTAINER_NAME = "vectorforge"
HEALTH_URL = "http://localhost:3001/health/live"
POLL_INTERVAL_SECONDS = 2
MAX_WAIT_SECONDS = 120


def _container_state() -> str | None:
    """Return the Docker container status string for CONTAINER_NAME.

    Returns:
      The status string (e.g. "running", "exited") or None if the container
      does not exist or the inspect command fails.
    """
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Status}}", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return None

    return result.stdout.strip()


def _is_api_live() -> bool:
    """Return True if the API responds with HTTP 200 on /health/live.

    Returns:
      True if the endpoint returns 200, False on any error or non-200 status.
    """
    try:
        resp = requests.get(HEALTH_URL, timeout=3)
        is_ok: bool = resp.status_code == 200
        return is_ok
    except requests.RequestException:
        return False


def ensure_container_ready() -> None:
    """Start the vectorforge container if needed and block until the API is live.

    Inspects the container state via docker inspect. If the container is not
    running, issues docker compose up -d. Polls /health/live every
    POLL_INTERVAL_SECONDS until a 200 response is received or MAX_WAIT_SECONDS
    elapses, in which case the process exits with a non-zero status.

    Raises:
      SystemExit: If docker compose up fails or the API does not come live in time.
    """
    state = _container_state()

    if state == "running":
        print(f"Container '{CONTAINER_NAME}' is already running. Waiting for API...")
    elif state in ("created", "restarting", "paused", "exited", "dead", None):
        if state is None:
            print(
                f"Container '{CONTAINER_NAME}' not found. Starting via docker compose..."
            )
        else:
            print(
                f"Container '{CONTAINER_NAME}' is '{state}'. Starting via docker compose..."
            )

        result = subprocess.run(
            ["docker", "compose", "up", "-d"],
            capture_output=False,
        )

        if result.returncode != 0:
            print("ERROR: 'docker compose up -d' failed. Is Docker running?")
            sys.exit(1)
    else:
        print(f"Container '{CONTAINER_NAME}' is in state '{state}'. Waiting for API...")

    elapsed = 0
    while elapsed < MAX_WAIT_SECONDS:
        if _is_api_live():
            print(f"API is live at {HEALTH_URL.rsplit('/health', 1)[0]}\n")
            return

        print(f"  Waiting for API... ({elapsed}s / {MAX_WAIT_SECONDS}s)")
        time.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS

    print(f"ERROR: API did not become ready within {MAX_WAIT_SECONDS}s. Aborting.")
    sys.exit(1)


_COLUMN_WIDTH = 26


def _print_help() -> None:
    """Print all available feature keys grouped by category."""
    print("\nAvailable features:\n")

    groups: dict[str, list[str]] = {}
    for key in FEATURES:
        group = key.split(":")[0]
        groups.setdefault(group, []).append(key)

    for group, keys in groups.items():
        for key in keys:
            _, description = FEATURES[key]
            print(f"  {key:<{_COLUMN_WIDTH}}  {description}")
        print()

    print(f"  {'help':<{_COLUMN_WIDTH}}  Show this list")
    print(f"  {'quit':<{_COLUMN_WIDTH}}  Exit the demo")
    print()


def _prompt_shutdown() -> None:
    """Ask the user what to do with the running Docker container on exit.

    Presents four options — keep it running, stop it, remove the container,
    or remove the container and all volume data. Options 3 and 4 require an
    explicit confirmation before proceeding.
    """
    print("\nWhat should happen to the Docker container?")
    print("  [1] Keep it running  (default)")
    print("  [2] Stop it          (docker compose stop)")
    print("  [3] Remove it        (docker compose down)")
    print("  [4] Remove it + data (docker compose down --volumes)")

    try:
        choice = input("  Choice [1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = ""

    if not choice or choice == "1":
        print(f"  Container '{CONTAINER_NAME}' left running.")
        return

    if choice == "2":
        print(f"  Stopping '{CONTAINER_NAME}'...")
        subprocess.run(["docker", "compose", "stop"], capture_output=False)
        return

    if choice == "3":
        try:
            confirm = (
                input("  Remove container? Data volume will be preserved. [y/N]: ")
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            confirm = ""

        if confirm not in ("y", "yes"):
            print("  Cancelled — container left running.")
            return

        print(f"  Removing '{CONTAINER_NAME}'...")
        subprocess.run(["docker", "compose", "down"], capture_output=False)
        return

    if choice == "4":
        try:
            confirm = (
                input(
                    "  Remove container and all volume data? This cannot be undone. [y/N]: "
                )
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            confirm = ""

        if confirm not in ("y", "yes"):
            print("  Cancelled — container left running.")
            return

        print(f"  Removing '{CONTAINER_NAME}' and volume data...")
        subprocess.run(["docker", "compose", "down", "--volumes"], capture_output=False)
        return

    print(f"  Unrecognised choice '{choice}' - container left running.")


def run_repl() -> None:
    """Display the feature menu and loop until the user quits."""
    print("=" * 60)
    print("  VectorForge Demo")
    print("=" * 60)
    _print_help()

    while True:
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            _prompt_shutdown()
            break

        if not raw:
            continue

        if raw in ("quit", "exit", "q"):
            _prompt_shutdown()
            break

        if raw in ("help", "h", "?"):
            _print_help()
            continue

        if raw not in FEATURES:
            print(f"  Unknown feature '{raw}'. Type 'help' to see available features.")
            continue

        handler, _ = FEATURES[raw]
        try:
            handler()
        except KeyboardInterrupt:
            print("\n  (cancelled)")
        except Exception as e:
            print(f"\n  ERROR: {e}\n")
