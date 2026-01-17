from pathlib import Path

import pytest

import docsearch.server
from docsearch.server import init_server, search


@pytest.fixture
def sample_file() -> Path:
    """Return path to existing test fixture."""
    return Path(__file__).parent / "fixtures" / "sample.md"


@pytest.fixture
def initialized_server(sample_file: Path, tmp_path: Path) -> None:
    """Initialize server with test data."""
    db_path = tmp_path / "test.db"
    init_server([sample_file], db_path)


class TestInitServer:
    def test_init_creates_db(self, sample_file: Path, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        init_server([sample_file], db_path)
        assert db_path.exists()

    def test_init_temp_db(self, sample_file: Path) -> None:
        # No db_path = temp file
        init_server([sample_file], None)
        # Should not raise, server should be queryable
        result = search("sample")
        assert "results" in result


class TestSearchTool:
    def test_search_returns_results(self, initialized_server: None) -> None:
        result = search("sample")
        assert "results" in result
        assert len(result["results"]) > 0

    def test_search_no_results(self, initialized_server: None) -> None:
        result = search("xyznonexistent123")
        assert result["results"] == []

    def test_search_not_initialized(self) -> None:
        # Reset module state
        docsearch.server._db_path = None
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
