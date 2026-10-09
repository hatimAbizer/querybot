"""Health check endpoint."""

from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Service health check")
def health() -> HealthResponse:
    """Return service status. Useful for readiness probes and smoke tests."""
    return HealthResponse(status="ok")
