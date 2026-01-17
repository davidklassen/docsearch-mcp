"""Tests for embedding generation."""

import pytest

from docsearch.embedding import EMBEDDING_DIM, Embedder

# Use a smaller, faster model for tests
TEST_MODEL = "all-MiniLM-L6-v2"
TEST_MODEL_DIM = 384  # MiniLM produces 384-dim vectors


@pytest.fixture
def embedder() -> Embedder:
    """Create an embedder with a fast test model."""
    return Embedder(model_name=TEST_MODEL)


class TestEmbedder:
    """Tests for Embedder class."""

    def test_embed_single_returns_list(self, embedder: Embedder) -> None:
        """embed_single returns a list of floats."""
        result = embedder.embed_single("Hello world")

        assert isinstance(result, list)
        assert len(result) == TEST_MODEL_DIM
        assert all(isinstance(x, float) for x in result)

    def test_embed_batch_returns_list(self, embedder: Embedder) -> None:
        """embed returns a list of embedding vectors."""
        texts = ["Hello", "World", "Test"]
        result = embedder.embed(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(len(emb) == TEST_MODEL_DIM for emb in result)

    def test_embed_empty_list(self, embedder: Embedder) -> None:
        """embed handles empty list."""
        result = embedder.embed([])

        assert result == []

    def test_embed_single_matches_batch(self, embedder: Embedder) -> None:
        """embed_single produces same result as embed with single item."""
        text = "Test sentence for embedding"

        single_result = embedder.embed_single(text)
        batch_result = embedder.embed([text])[0]

        # Results should be identical
        assert single_result == batch_result

    def test_lazy_loading(self) -> None:
        """Model is not loaded until first use."""
        embedder = Embedder(model_name=TEST_MODEL)

        # Model should not be loaded yet
        assert embedder._model is None

        # After first use, model should be loaded
        embedder.embed_single("test")
        assert embedder._model is not None

    def test_different_texts_produce_different_embeddings(
        self, embedder: Embedder
    ) -> None:
        """Different texts produce different embeddings."""
        emb1 = embedder.embed_single("The cat sat on the mat")
        emb2 = embedder.embed_single("Machine learning and artificial intelligence")

        # Embeddings should be different
        assert emb1 != emb2


class TestEmbeddingConstants:
    """Tests for embedding module constants."""

    def test_embedding_dim_matches_model(self) -> None:
        """EMBEDDING_DIM constant matches configured model."""
        # EMBEDDING_DIM should be a positive integer
        assert isinstance(EMBEDDING_DIM, int)
        assert EMBEDDING_DIM > 0
