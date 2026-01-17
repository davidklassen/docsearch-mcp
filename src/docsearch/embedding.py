"""Embedding generation for vector search."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

import os

# Default model - can be overridden via DOCSEARCH_EMBEDDING_MODEL env var
# Options:
#   - "google/embeddinggemma-300M" (768-dim, requires HF auth)
#   - "BAAI/bge-small-en-v1.5" (384-dim, no auth required)
#   - "all-MiniLM-L6-v2" (384-dim, no auth required)
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL = os.environ.get("DOCSEARCH_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

# Dimension lookup for supported models
_MODEL_DIMS = {
    "google/embeddinggemma-300M": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "all-MiniLM-L6-v2": 384,
}
EMBEDDING_DIM = _MODEL_DIMS.get(EMBEDDING_MODEL, 384)


class Embedder:
    """Generate embeddings using sentence-transformers.

    Model is loaded lazily on first use.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        """Initialize embedder with model name.

        Args:
            model_name: HuggingFace model name for embeddings
        """
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        """Load model lazily on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (each is list of floats)
        """
        if not texts:
            return []

        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        results = self.embed([text])
        return results[0]
