import subprocess
from pathlib import Path

from docsearch.cli import expand_globs


class TestServeCommand:
    def test_serve_help(self) -> None:
        result = subprocess.run(
            ["uv", "run", "docsearch", "serve", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "MCP server" in result.stdout or "stdio" in result.stdout
        assert "--db" in result.stdout

    def test_serve_requires_files(self) -> None:
        result = subprocess.run(
            ["uv", "run", "docsearch", "serve"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0  # Should fail without files


class TestExpandGlobs:
    def test_expand_single_file(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "sample.md"
        result = expand_globs((str(fixture),))
        assert result == [fixture]

    def test_expand_glob_pattern(self) -> None:
        fixtures_dir = Path(__file__).parent / "fixtures"
        pattern = str(fixtures_dir / "*.md")
        result = expand_globs((pattern,))
        assert len(result) >= 1
        assert all(f.suffix == ".md" for f in result)

    def test_expand_recursive_glob(self) -> None:
        tests_dir = Path(__file__).parent
        pattern = str(tests_dir / "**" / "*.md")
        result = expand_globs((pattern,))
        # Should find sample.md and sample2.md in fixtures/
        assert len(result) >= 2
        assert all(f.suffix == ".md" for f in result)

    def test_expand_no_matches(self) -> None:
        result = expand_globs(("nonexistent/**/*.xyz",))
        assert result == []

    def test_expand_multiple_patterns(self) -> None:
        fixtures_dir = Path(__file__).parent / "fixtures"
        pattern1 = str(fixtures_dir / "sample.md")
        pattern2 = str(fixtures_dir / "sample2.md")
        result = expand_globs((pattern1, pattern2))
        assert len(result) == 2
