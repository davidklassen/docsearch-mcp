import json
from pathlib import Path

from docsearch.chunker import chunk_file

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_chunk_file_sample() -> None:
    sample_path = FIXTURES_DIR / "sample.md"
    chunks = list(chunk_file(sample_path))

    # Should have 5 sections: H1, H2 Section 1, H3 1.1, H3 1.2, H2 Section 2
    assert len(chunks) == 5

    # Check document title extraction
    assert all(c.source.name == "Sample Document" for c in chunks)
    assert all(c.source.id == "sample-document" for c in chunks)

    # Check hierarchy
    root = chunks[0]
    assert root.section_id == "sample-document"
    assert root.parent_id is None
    assert root.title == "Sample Document"

    section1 = chunks[1]
    assert section1.parent_id == "sample-document"
    assert "Section 1" in section1.title

    section1_1 = chunks[2]
    assert section1_1.parent_id == "sample-document--section-1"

    section2 = chunks[4]
    assert section2.parent_id == "sample-document"
    assert "Section 2" in section2.title


def test_chunk_file_paths() -> None:
    sample_path = FIXTURES_DIR / "sample.md"
    chunks = list(chunk_file(sample_path))

    # Check breadcrumb paths
    assert chunks[0].path == "Sample Document"
    assert chunks[1].path == "Sample Document → Section 1"
    assert chunks[2].path == "Sample Document → Section 1 → Section 1.1"


def test_chunk_file_code_block_preserved() -> None:
    sample_path = FIXTURES_DIR / "sample.md"
    chunks = list(chunk_file(sample_path))

    section2 = chunks[4]
    assert "```python" in section2.content
    assert "def example():" in section2.content


def test_chunk_file_headers_in_content() -> None:
    """LangChain includes headers in content when strip_headers=False."""
    sample_path = FIXTURES_DIR / "sample.md"
    chunks = list(chunk_file(sample_path))

    # Headers should be in content
    assert "# Sample Document" in chunks[0].content
    assert "## Section 1" in chunks[1].content
    assert "### Section 1.1" in chunks[2].content


def test_chunk_to_json() -> None:
    sample_path = FIXTURES_DIR / "sample.md"
    chunks = list(chunk_file(sample_path))

    # Verify JSON serialization works
    for chunk in chunks:
        json_str = json.dumps(chunk.to_dict())
        parsed = json.loads(json_str)

        assert parsed["section_id"] == chunk.section_id
        assert parsed["source"]["id"] == chunk.source.id
