"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Return a TestClient for the FastAPI app."""
    return TestClient(app)
