"""Tests for database operations."""

from pathlib import Path

import pytest

from docsearch.db import (
    DatabaseNotFoundError,
    create_schema,
    delete_source,
    index_chunks,
    open_db,
    search,
)
from docsearch.models import Chunk, Source


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Temporary database path."""
    return tmp_path / "test.db"


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Sample chunks for testing."""
    source = Source(
        id="test-document",
        name="Test Document",
        file="test.md",
        lines=(1, 100),
    )
    return [
        Chunk(
            section_id="test-document--section-1",
            parent_id=None,  # Top-level section
            title="Section 1",
            path="Test Document → Section 1",
            content="This is the first section with some content about vectors.",
            source=source,
        ),
        Chunk(
            section_id="test-document--section-2",
            parent_id=None,  # Top-level section
            title="Section 2",
            path="Test Document → Section 2",
            content="This is the second section about exception handling.",
            source=source,
        ),
        Chunk(
            section_id="test-document--section-1--subsection",
            parent_id="test-document--section-1",
            title="Subsection 1.1",
            path="Test Document → Section 1 → Subsection 1.1",
            content="Detailed content about vectors and memory alignment.",
            source=source,
        ),
    ]


@pytest.fixture
def sample_chunks_2() -> list[Chunk]:
    """Second set of sample chunks for multi-document testing."""
    source = Source(
        id="second-document",
        name="Second Document",
        file="second.md",
        lines=(1, 50),
    )
    return [
        Chunk(
            section_id="second-document--intro",
            parent_id=None,
            title="Introduction",
            path="Second Document → Introduction",
            content="An introduction to ARM architecture.",
            source=source,
        ),
        Chunk(
            section_id="second-document--registers",
            parent_id=None,
            title="Registers",
            path="Second Document → Registers",
            content="Information about VBAR_EL1 and other registers.",
            source=source,
        ),
    ]


class TestCreateSchema:
    """Tests for schema creation."""

    def test_create_schema_creates_tables(self, db_path: Path) -> None:
        """Verify all tables exist after schema creation."""
        import sqlite3

        conn = sqlite3.connect(db_path)
        create_schema(conn)

        # Check tables exist
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor]

        assert "sources" in tables
        assert "sections" in tables
        conn.close()

    def test_create_schema_creates_fts(self, db_path: Path) -> None:
        """Verify FTS table exists."""
        import sqlite3

        conn = sqlite3.connect(db_path)
        create_schema(conn)

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sections_fts'"
        )
        assert cursor.fetchone() is not None
        conn.close()

    def test_create_schema_creates_triggers(self, db_path: Path) -> None:
        """Verify triggers exist."""
        import sqlite3

        conn = sqlite3.connect(db_path)
        create_schema(conn)

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' ORDER BY name"
        )
        triggers = [row[0] for row in cursor]

        assert "sections_ai" in triggers  # after insert
        assert "sections_ad" in triggers  # after delete
        assert "sections_au" in triggers  # after update
        conn.close()

    def test_create_schema_idempotent(self, db_path: Path) -> None:
        """Running schema creation twice doesn't error."""
        import sqlite3

        conn = sqlite3.connect(db_path)
        create_schema(conn)
        create_schema(conn)  # Should not raise
        conn.close()


class TestOpenDb:
    """Tests for database connection management."""

    def test_open_db_creates_if_requested(self, db_path: Path) -> None:
        """Create flag creates DB and schema."""
        assert not db_path.exists()

        with open_db(db_path, create=True) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor]
            assert "sources" in tables

        assert db_path.exists()

    def test_open_db_not_found(self, db_path: Path) -> None:
        """Raises DatabaseNotFoundError when DB missing."""
        with pytest.raises(DatabaseNotFoundError):
            with open_db(db_path, create=False):
                pass


class TestIndexChunks:
    """Tests for chunk indexing."""

    def test_index_chunks_single_file(
        self, db_path: Path, sample_chunks: list[Chunk]
    ) -> None:
        """Index sample chunks and verify records."""
        with open_db(db_path, create=True) as conn:
            count = index_chunks(conn, sample_chunks)

            assert count == 3

            # Verify source was created
            cursor = conn.execute(
                "SELECT * FROM sources WHERE id = ?", ("test-document",)
            )
            source_row = cursor.fetchone()
            assert source_row is not None
            assert source_row["name"] == "Test Document"

            # Verify sections were created
            cursor = conn.execute("SELECT COUNT(*) FROM sections")
            assert cursor.fetchone()[0] == 3

    def test_index_chunks_multiple_sources(
        self,
        db_path: Path,
        sample_chunks: list[Chunk],
        sample_chunks_2: list[Chunk],
    ) -> None:
        """Index multiple documents, verify separation."""
        with open_db(db_path, create=True) as conn:
            count1 = index_chunks(conn, sample_chunks)
            count2 = index_chunks(conn, sample_chunks_2)

            assert count1 == 3
            assert count2 == 2

            # Verify both sources exist
            cursor = conn.execute("SELECT COUNT(*) FROM sources")
            assert cursor.fetchone()[0] == 2

            # Verify sections from both exist
            cursor = conn.execute("SELECT COUNT(*) FROM sections")
            assert cursor.fetchone()[0] == 5

    def test_index_chunks_with_same_section_ids(self, db_path: Path) -> None:
        """Two sources with identical section_ids can both be indexed."""
        source1 = Source(
            id="doc-alpha",
            name="Document Alpha",
            file="alpha.md",
            lines=(1, 50),
        )
        source2 = Source(
            id="doc-beta",
            name="Document Beta",
            file="beta.md",
            lines=(1, 50),
        )

        # Both documents have sections with the same section_id
        chunks1 = [
            Chunk(
                section_id="introduction",
                parent_id=None,
                title="Introduction",
                path="Document Alpha → Introduction",
                content="Introduction to Alpha concepts.",
                source=source1,
            ),
            Chunk(
                section_id="overview",
                parent_id=None,
                title="Overview",
                path="Document Alpha → Overview",
                content="Overview of Alpha features.",
                source=source1,
            ),
        ]
        chunks2 = [
            Chunk(
                section_id="introduction",  # Same section_id as chunks1
                parent_id=None,
                title="Introduction",
                path="Document Beta → Introduction",
                content="Introduction to Beta concepts.",
                source=source2,
            ),
            Chunk(
                section_id="overview",  # Same section_id as chunks1
                parent_id=None,
                title="Overview",
                path="Document Beta → Overview",
                content="Overview of Beta features.",
                source=source2,
            ),
        ]

        with open_db(db_path, create=True) as conn:
            # Index both - should NOT raise UNIQUE constraint error
            count1 = index_chunks(conn, chunks1)
            count2 = index_chunks(conn, chunks2)

            assert count1 == 2
            assert count2 == 2

            # Verify all 4 sections exist
            cursor = conn.execute("SELECT COUNT(*) FROM sections")
            assert cursor.fetchone()[0] == 4

            # Verify we can search and get results from both sources
            results = search(conn, "introduction")
            source_ids = {r.source.id for r in results}
            assert "doc-alpha" in source_ids
            assert "doc-beta" in source_ids

    def test_index_chunks_reindex_replaces(
        self, db_path: Path, sample_chunks: list[Chunk]
    ) -> None:
        """Re-indexing same file replaces data."""
        with open_db(db_path, create=True) as conn:
            index_chunks(conn, sample_chunks)

            # Modify and re-index
            modified_chunks = [
                Chunk(
                    section_id="test-document--new-section",
                    parent_id=None,
                    title="New Section",
                    path="Test Document → New Section",
                    content="Completely new content.",
                    source=sample_chunks[0].source,
                )
            ]
            count = index_chunks(conn, modified_chunks)

            assert count == 1

            # Verify old sections are gone
            cursor = conn.execute("SELECT COUNT(*) FROM sections")
            assert cursor.fetchone()[0] == 1

            # Verify new section exists
            cursor = conn.execute(
                "SELECT id FROM sections WHERE id = ?", ("test-document--new-section",)
            )
            assert cursor.fetchone() is not None

    def test_index_chunks_fts_populated(
        self, db_path: Path, sample_chunks: list[Chunk]
    ) -> None:
        """Verify FTS table has content after indexing."""
        with open_db(db_path, create=True) as conn:
            index_chunks(conn, sample_chunks)

            # Query FTS directly
            cursor = conn.execute("SELECT COUNT(*) FROM sections_fts")
            assert cursor.fetchone()[0] == 3

    def test_index_chunks_empty_list(self, db_path: Path) -> None:
        """Indexing empty list succeeds with count 0."""
        with open_db(db_path, create=True) as conn:
            count = index_chunks(conn, [])
            assert count == 0


class TestDeleteSource:
    """Tests for source deletion."""

    def test_delete_source_removes_sections(
        self, db_path: Path, sample_chunks: list[Chunk]
    ) -> None:
        """Deleting source removes all its sections."""
        with open_db(db_path, create=True) as conn:
            index_chunks(conn, sample_chunks)

            delete_source(conn, "test-document")
            conn.commit()

            # Verify source is gone
            cursor = conn.execute("SELECT COUNT(*) FROM sources")
            assert cursor.fetchone()[0] == 0

            # Verify sections are gone
            cursor = conn.execute("SELECT COUNT(*) FROM sections")
            assert cursor.fetchone()[0] == 0


class TestSearch:
    """Tests for search functionality."""

    def test_search_returns_results(
        self, db_path: Path, sample_chunks: list[Chunk]
    ) -> None:
        """Basic search returns matches."""
        with open_db(db_path, create=True) as conn:
            index_chunks(conn, sample_chunks)

            results = search(conn, "vectors")

            assert len(results) >= 1
            # At least one result should mention vectors
            assert any("vectors" in r.content.lower() for r in results)

    def test_search_bm25_ranking(
        self, db_path: Path, sample_chunks: list[Chunk]
    ) -> None:
        """Results are ordered by relevance."""
        with open_db(db_path, create=True) as conn:
            index_chunks(conn, sample_chunks)

            results = search(conn, "vectors")

            # Results should have scores
            assert all(r.score > 0 for r in results)

            # Results should be ordered by score (descending)
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_search_with_limit(self, db_path: Path, sample_chunks: list[Chunk]) -> None:
        """Limit parameter respected."""
        with open_db(db_path, create=True) as conn:
            index_chunks(conn, sample_chunks)

            results = search(conn, "section", limit=1)

            assert len(results) <= 1

    def test_search_with_source_filter(
        self,
        db_path: Path,
        sample_chunks: list[Chunk],
        sample_chunks_2: list[Chunk],
    ) -> None:
        """Source filtering works."""
        with open_db(db_path, create=True) as conn:
            index_chunks(conn, sample_chunks)
            index_chunks(conn, sample_chunks_2)

            # Search with filter
            results = search(conn, "section", source_id="test-document")

            # All results should be from the filtered source
            assert all(r.source.id == "test-document" for r in results)

    def test_search_returns_results_from_multiple_sources(self, db_path: Path) -> None:
        """Search returns results from multiple sources when both match."""
        # Create two sources with overlapping content
        source1 = Source(
            id="doc-one",
            name="Document One",
            file="doc1.md",
            lines=(1, 50),
        )
        source2 = Source(
            id="doc-two",
            name="Document Two",
            file="doc2.md",
            lines=(1, 50),
        )

        chunks1 = [
            Chunk(
                section_id="doc-one--exceptions",
                parent_id=None,
                title="Exception Handling",
                path="Document One → Exception Handling",
                content="This section covers exception vectors and error handling.",
                source=source1,
            ),
        ]
        chunks2 = [
            Chunk(
                section_id="doc-two--vectors",
                parent_id=None,
                title="Vector Tables",
                path="Document Two → Vector Tables",
                content="Exception vectors are stored in memory-aligned tables.",
                source=source2,
            ),
        ]

        with open_db(db_path, create=True) as conn:
            index_chunks(conn, chunks1)
            index_chunks(conn, chunks2)

            # Search for term that appears in both documents
            results = search(conn, "exception vectors", limit=10)

            # Should have results from both sources
            source_ids = {r.source.id for r in results}
            assert "doc-one" in source_ids, "Missing results from doc-one"
            assert "doc-two" in source_ids, "Missing results from doc-two"

    def test_search_no_results(self, db_path: Path, sample_chunks: list[Chunk]) -> None:
        """Non-matching query returns empty list."""
        with open_db(db_path, create=True) as conn:
            index_chunks(conn, sample_chunks)

            results = search(conn, "xyznonexistent")

            assert results == []

    def test_search_result_structure(
        self, db_path: Path, sample_chunks: list[Chunk]
    ) -> None:
        """Search results have correct structure."""
        with open_db(db_path, create=True) as conn:
            index_chunks(conn, sample_chunks)

            results = search(conn, "vectors")

            assert len(results) > 0
            result = results[0]

            # Check all fields are present
            assert result.section_id is not None
            assert result.title is not None
            assert result.path is not None
            assert result.content is not None
            assert result.source is not None
            assert result.score is not None

            # Check source structure
            assert result.source.id is not None
            assert result.source.name is not None
            assert result.source.file is not None

    def test_search_to_dict(self, db_path: Path, sample_chunks: list[Chunk]) -> None:
        """Search results serialize correctly."""
        with open_db(db_path, create=True) as conn:
            index_chunks(conn, sample_chunks)

            results = search(conn, "vectors")

            assert len(results) > 0
            result_dict = results[0].to_dict()

            assert "section_id" in result_dict
            assert "score" in result_dict
            assert "source" in result_dict
            assert isinstance(result_dict["source"], dict)
