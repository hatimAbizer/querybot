"""Ollama generation client: call the local LLM and return the raw text response."""

from __future__ import annotations

import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# Timeout for generation requests; generous to handle slow local inference
GENERATION_TIMEOUT = 240.0

# Ollama generation options to constrain output length and reduce context load.
# num_predict caps output tokens; num_ctx limits the model's context window.
_OLLAMA_OPTIONS = {
    "num_predict": 512,
    "num_ctx": 2048,
    "temperature": 0.1,
}


def generate(prompt: str) -> str:
    """Send a prompt to Ollama and return the generated text.

    Raises:
        httpx.HTTPError: If the Ollama API is unreachable or returns an error.
    """
    settings = get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"

    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": _OLLAMA_OPTIONS,
    }

    logger.info("Requesting generation from Ollama model '%s'", settings.ollama_model)
    try:
        with httpx.Client(timeout=GENERATION_TIMEOUT) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {settings.ollama_base_url}. "
            "Make sure Ollama is running (`ollama serve`)."
        ) from exc
    except httpx.HTTPStatusError as exc:
        msg = ""
        try:
            msg = exc.response.json().get("error", "")
        except Exception:
            msg = exc.response.text
        raise RuntimeError(
            f"Ollama error ({exc.response.status_code}): {msg or str(exc)}"
        ) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    data = response.json()
    text: str = data.get("response", "").strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")

    logger.info("Generation complete (%d characters)", len(text))
    return text


def is_ollama_available() -> bool:
    """Return True if the Ollama API responds to a health check."""
    settings = get_settings()
    url = f"{settings.ollama_base_url.rstrip('/')}/api/tags"
    try:
        with httpx.Client(timeout=5.0) as client:
            r = client.get(url)
            return r.status_code == 200
    except Exception:
        return False
