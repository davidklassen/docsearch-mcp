"""Tests for the index CLI command."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from docsearch.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_file() -> Path:
    """Path to sample markdown file."""
    return Path("tests/fixtures/sample.md")


class TestIndexCommand:
    """Tests for the index CLI command."""

    def test_index_file_creates_db(
        self, runner: CliRunner, sample_file: Path, tmp_path: Path
    ) -> None:
        """Running on file creates database."""
        db_path = tmp_path / "test.db"
        assert not db_path.exists()

        result = runner.invoke(cli, ["index", str(sample_file), "--db", str(db_path)])

        assert result.exit_code == 0
        assert db_path.exists()

    def test_index_file_outputs_summary(
        self, runner: CliRunner, sample_file: Path, tmp_path: Path
    ) -> None:
        """Correct output message after indexing."""
        db_path = tmp_path / "test.db"

        result = runner.invoke(cli, ["index", str(sample_file), "--db", str(db_path)])

        assert result.exit_code == 0
        assert "Indexed" in result.output
        assert "chunks" in result.output
        assert "Sample Document" in result.output

    def test_index_file_db_option(
        self, runner: CliRunner, sample_file: Path, tmp_path: Path
    ) -> None:
        """Custom DB path works."""
        custom_db = tmp_path / "custom" / "path" / "mydb.db"
        custom_db.parent.mkdir(parents=True)

        result = runner.invoke(cli, ["index", str(sample_file), "--db", str(custom_db)])

        assert result.exit_code == 0
        assert custom_db.exists()

    def test_index_stdin_jsonl(
        self, runner: CliRunner, sample_file: Path, tmp_path: Path
    ) -> None:
        """Piped JSONL is indexed."""
        db_path = tmp_path / "test.db"

        # First generate JSONL from chunks command
        chunks_result = runner.invoke(cli, ["chunks", str(sample_file)])
        assert chunks_result.exit_code == 0

        # Then pipe to index
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
        self, runner: CliRunner, sample_file: Path, tmp_path: Path
    ) -> None:
        """Re-indexing same file replaces data."""
        db_path = tmp_path / "test.db"

        # Index twice
        runner.invoke(cli, ["index", str(sample_file), "--db", str(db_path)])
        result = runner.invoke(cli, ["index", str(sample_file), "--db", str(db_path)])

        assert result.exit_code == 0
        assert "Indexed" in result.output

        # Search should still work
        search_result = runner.invoke(cli, ["search", "section", "--db", str(db_path)])
        assert search_result.exit_code == 0

    def test_index_multiple_files_pipeline(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Multiple files can be indexed via pipeline."""
        db_path = tmp_path / "test.db"

        # Get chunks from sample.md
        chunks_result = runner.invoke(cli, ["chunks", "tests/fixtures/sample.md"])
        assert chunks_result.exit_code == 0

        # Index via stdin
        result = runner.invoke(
            cli, ["index", "--db", str(db_path)], input=chunks_result.output
        )

        assert result.exit_code == 0
        assert db_path.exists()
