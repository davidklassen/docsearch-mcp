"""Tests for the search CLI command."""

import json
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


@pytest.fixture
def indexed_db(runner: CliRunner, sample_file: Path, tmp_path: Path) -> Path:
    """Database with sample data indexed."""
    db_path = tmp_path / "test.db"
    result = runner.invoke(cli, ["index", str(sample_file), "--db", str(db_path)])
    assert result.exit_code == 0
    return db_path


class TestSearchCommand:
    """Tests for the search CLI command."""

    def test_search_returns_jsonl(self, runner: CliRunner, indexed_db: Path) -> None:
        """Output is valid JSONL."""
        result = runner.invoke(cli, ["search", "section", "--db", str(indexed_db)])

        assert result.exit_code == 0
        # Each non-empty line should be valid JSON
        for line in result.output.strip().split("\n"):
            if line:
                data = json.loads(line)
                assert isinstance(data, dict)

    def test_search_includes_score(self, runner: CliRunner, indexed_db: Path) -> None:
        """Each result has score field."""
        result = runner.invoke(cli, ["search", "section", "--db", str(indexed_db)])

        assert result.exit_code == 0
        for line in result.output.strip().split("\n"):
            if line:
                data = json.loads(line)
                assert "score" in data
                assert isinstance(data["score"], float)

    def test_search_limit_option(self, runner: CliRunner, indexed_db: Path) -> None:
        """--limit option works."""
        result = runner.invoke(
            cli, ["search", "section", "--db", str(indexed_db), "--limit", "2"]
        )

        assert result.exit_code == 0
        lines = [ln for ln in result.output.strip().split("\n") if ln]
        assert len(lines) <= 2

    def test_search_limit_short_option(
        self, runner: CliRunner, indexed_db: Path
    ) -> None:
        """-n short option for limit works."""
        result = runner.invoke(
            cli, ["search", "section", "--db", str(indexed_db), "-n", "1"]
        )

        assert result.exit_code == 0
        lines = [ln for ln in result.output.strip().split("\n") if ln]
        assert len(lines) <= 1

    def test_search_no_db_error(self, runner: CliRunner, tmp_path: Path) -> None:
        """Helpful error when DB missing."""
        nonexistent_db = tmp_path / "nonexistent.db"

        result = runner.invoke(cli, ["search", "test", "--db", str(nonexistent_db)])

        assert result.exit_code != 0
        assert "Database not found" in result.output
        assert "docsearch index" in result.output

    def test_search_empty_results(self, runner: CliRunner, indexed_db: Path) -> None:
        """No results for non-matching query."""
        result = runner.invoke(
            cli, ["search", "xyznonexistent", "--db", str(indexed_db)]
        )

        assert result.exit_code == 0
        assert result.output.strip() == ""

    def test_search_result_structure(self, runner: CliRunner, indexed_db: Path) -> None:
        """Results have correct structure."""
        result = runner.invoke(cli, ["search", "section", "--db", str(indexed_db)])

        assert result.exit_code == 0
        lines = [ln for ln in result.output.strip().split("\n") if ln]
        assert len(lines) > 0

        data = json.loads(lines[0])
        assert "section_id" in data
        assert "title" in data
        assert "path" in data
        assert "content" in data
        assert "source" in data
        assert "score" in data

        # Check source structure
        assert "id" in data["source"]
        assert "name" in data["source"]
        assert "file" in data["source"]

    def test_search_source_filter(self, runner: CliRunner, indexed_db: Path) -> None:
        """--source filter option works."""
        result = runner.invoke(
            cli,
            [
                "search",
                "section",
                "--db",
                str(indexed_db),
                "--source",
                "sample-document",
            ],
        )

        assert result.exit_code == 0
        for line in result.output.strip().split("\n"):
            if line:
                data = json.loads(line)
                assert data["source"]["id"] == "sample-document"

    def test_search_source_filter_short(
        self, runner: CliRunner, indexed_db: Path
    ) -> None:
        """-s short option for source filter works."""
        result = runner.invoke(
            cli,
            ["search", "section", "--db", str(indexed_db), "-s", "sample-document"],
        )

        assert result.exit_code == 0
