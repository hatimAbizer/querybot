"""Embedding model wrapper using sentence-transformers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model(model_name: str):
    """Load and cache the embedding model (expensive, do once per process)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embeddings. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Embedding model loaded.")
    return model


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Return a list of embedding vectors for the given texts.

    Inputs longer than the model's token limit (256 word-pieces for
    all-MiniLM-L6-v2) are silently truncated by the model. Keep chunks
    appropriately sized.
    """
    settings = get_settings()
    model = _load_model(settings.embedding_model)
    vectors = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(query: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
