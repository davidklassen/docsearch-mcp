from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import docsearch.server
from docsearch.embedding import EMBEDDING_DIM
from docsearch.server import init_server, search


@pytest.fixture
def sample_file() -> Path:
    """Return path to existing test fixture."""
    return Path(__file__).parent / "fixtures" / "sample.md"


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock Embedder that returns fake embeddings."""
    embedder = MagicMock()
    # Return embeddings matching input list length
    embedder.embed.side_effect = lambda texts: [[0.1] * EMBEDDING_DIM for _ in texts]
    embedder.embed_single.return_value = [0.1] * EMBEDDING_DIM
    return embedder


@pytest.fixture
def initialized_server(
    sample_file: Path, tmp_path: Path, mock_embedder: MagicMock
) -> None:
    """Initialize server with test data."""
    db_path = tmp_path / "test.db"
    with patch("docsearch.embedding.Embedder", return_value=mock_embedder):
        init_server([sample_file], db_path)
    # Set the mock embedder on the module so search() can use it
    docsearch.server._embedder = mock_embedder


class TestInitServer:
    def test_init_creates_db(
        self, sample_file: Path, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        db_path = tmp_path / "test.db"
        with patch("docsearch.embedding.Embedder", return_value=mock_embedder):
            init_server([sample_file], db_path)
        assert db_path.exists()

    def test_init_temp_db(self, sample_file: Path, mock_embedder: MagicMock) -> None:
        # No db_path = temp file
        with patch("docsearch.embedding.Embedder", return_value=mock_embedder):
            init_server([sample_file], None)
        # Set the mock embedder on the module so search() can use it
        docsearch.server._embedder = mock_embedder
        # Should not raise, server should be queryable
        result = search("sample")
        assert "results" in result


class TestSearchTool:
    def test_search_returns_results(self, initialized_server: None) -> None:
        result = search("sample")
        assert "results" in result
        assert len(result["results"]) > 0

    def test_search_hybrid_returns_results(self, initialized_server: None) -> None:
        """Hybrid search returns results even for non-matching literal queries.

        This is expected behavior: while BM25 may find nothing,
        vector similarity search can still find related documents.
        """
        result = search("xyznonexistent123")
        # Hybrid search returns results via vector similarity
        assert "results" in result
        assert len(result["results"]) > 0

    def test_search_not_initialized(self) -> None:
        # Reset module state
        docsearch.server._db_path = None
        docsearch.server._embedder = None
        result = search("test")
        assert "error" in result

    def test_search_result_structure(self, initialized_server: None) -> None:
        result = search("sample")
        r = result["results"][0]
        assert "section_id" in r
        assert "parent_id" in r
        assert "title" in r
        assert "path" in r
        assert "content" in r
        assert "source" in r
        assert "score" in r
        assert isinstance(r["score"], float)
