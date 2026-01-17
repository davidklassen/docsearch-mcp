"""Tests for the index CLI command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from docsearch.cli import cli
from docsearch.embedding import EMBEDDING_DIM


@pytest.fixture
def runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_file() -> Path:
    """Path to sample markdown file."""
    return Path("tests/fixtures/sample.md")


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Mock Embedder that returns fake embeddings."""
    embedder = MagicMock()
    # Return embeddings matching input list length
    embedder.embed.side_effect = lambda texts: [[0.1] * EMBEDDING_DIM for _ in texts]
    embedder.embed_single.return_value = [0.1] * EMBEDDING_DIM
    return embedder


class TestIndexCommand:
    """Tests for the index CLI command."""

    def test_index_file_creates_db(
        self,
        runner: CliRunner,
        sample_file: Path,
        tmp_path: Path,
        mock_embedder: MagicMock,
    ) -> None:
        """Running on file creates database."""
        db_path = tmp_path / "test.db"
        assert not db_path.exists()

        with patch("docsearch.embedding.Embedder", return_value=mock_embedder):
            result = runner.invoke(
                cli, ["index", str(sample_file), "--db", str(db_path)]
            )

        assert result.exit_code == 0
        assert db_path.exists()

    def test_index_file_outputs_summary(
        self,
        runner: CliRunner,
        sample_file: Path,
        tmp_path: Path,
        mock_embedder: MagicMock,
    ) -> None:
        """Correct output message after indexing."""
        db_path = tmp_path / "test.db"

        with patch("docsearch.embedding.Embedder", return_value=mock_embedder):
            result = runner.invoke(
                cli, ["index", str(sample_file), "--db", str(db_path)]
            )

        assert result.exit_code == 0
        assert "Indexed" in result.output
        assert "chunks" in result.output
        assert "sample.md" in result.output
        assert "Total:" in result.output

    def test_index_file_db_option(
        self,
        runner: CliRunner,
        sample_file: Path,
        tmp_path: Path,
        mock_embedder: MagicMock,
    ) -> None:
        """Custom DB path works."""
        custom_db = tmp_path / "custom" / "path" / "mydb.db"
        custom_db.parent.mkdir(parents=True)

        with patch("docsearch.embedding.Embedder", return_value=mock_embedder):
            result = runner.invoke(
                cli, ["index", str(sample_file), "--db", str(custom_db)]
            )

        assert result.exit_code == 0
        assert custom_db.exists()

    def test_index_stdin_jsonl(
        self,
        runner: CliRunner,
        sample_file: Path,
        tmp_path: Path,
        mock_embedder: MagicMock,
    ) -> None:
        """Piped JSONL is indexed."""
        db_path = tmp_path / "test.db"

        # First generate JSONL from chunks command
        chunks_result = runner.invoke(cli, ["chunks", str(sample_file)])
        assert chunks_result.exit_code == 0

        # Then pipe to index
        with patch("docsearch.embedding.Embedder", return_value=mock_embedder):
            result = runner.invoke(
                cli, ["index", "--db", str(db_path)], input=chunks_result.output
            )

        assert result.exit_code == 0
        assert "Indexed" in result.output
        assert db_path.exists()

    def test_index_nonexistent_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Error message for missing file."""
        db_path = tmp_path / "test.db"

        result = runner.invoke(
            cli, ["index", "/nonexistent/file.md", "--db", str(db_path)]
        )

        assert result.exit_code != 0

    def test_index_reindex_replaces(
        self,
        runner: CliRunner,
        sample_file: Path,
        tmp_path: Path,
        mock_embedder: MagicMock,
    ) -> None:
        """Re-indexing same file replaces data."""
        db_path = tmp_path / "test.db"

        with patch("docsearch.embedding.Embedder", return_value=mock_embedder):
            # Index twice
            runner.invoke(cli, ["index", str(sample_file), "--db", str(db_path)])
            result = runner.invoke(
                cli, ["index", str(sample_file), "--db", str(db_path)]
            )

        assert result.exit_code == 0
        assert "Indexed" in result.output

        # Search should still work (mock the search embedder too)
        with patch("docsearch.embedding.Embedder", return_value=mock_embedder):
            search_result = runner.invoke(
                cli, ["search", "section", "--db", str(db_path)]
            )
        assert search_result.exit_code == 0

    def test_index_multiple_files_pipeline(
        self, runner: CliRunner, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """Multiple files can be indexed via pipeline."""
        db_path = tmp_path / "test.db"

        # Get chunks from sample.md
        chunks_result = runner.invoke(cli, ["chunks", "tests/fixtures/sample.md"])
        assert chunks_result.exit_code == 0

        # Index via stdin
        with patch("docsearch.embedding.Embedder", return_value=mock_embedder):
            result = runner.invoke(
                cli, ["index", "--db", str(db_path)], input=chunks_result.output
            )

        assert result.exit_code == 0
        assert db_path.exists()
