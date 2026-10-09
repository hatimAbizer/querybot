"""Tests for the health endpoint."""

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_health_schema(client: TestClient) -> None:
    """Health response must conform to HealthResponse schema."""
    from app.schemas import HealthResponse

    response = client.get("/health")
    parsed = HealthResponse(**response.json())
    assert parsed.status == "ok"
