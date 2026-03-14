"""System / health feature handlers for the VectorForge demo."""

from demo import client


def health() -> None:
    """GET /health: basic health check."""
    print("\n-- Health --")
    resp = client.get("/health")
    client.print_response(resp)


def health_ready() -> None:
    """GET /health/ready: readiness probe."""
    print("\n-- Health: Ready --")
    resp = client.get("/health/ready")
    client.print_response(resp)


def health_live() -> None:
    """GET /health/live: liveness probe."""
    print("\n-- Health: Live --")
    resp = client.get("/health/live")
    client.print_response(resp)


def metrics() -> None:
    """GET /collections/{name}/metrics: comprehensive collection metrics."""
    print("\n-- Collection Metrics --")
    collection_name = client.prompt("Collection name", default="vectorforge")
    resp = client.get(f"/collections/{collection_name}/metrics")
    client.print_response(resp)
