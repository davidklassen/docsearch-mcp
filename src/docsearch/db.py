"""Database operations for docsearch."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from docsearch.models import Chunk, SearchResult

DEFAULT_DB_PATH = Path("./docsearch.db")


class DatabaseNotFoundError(Exception):
    """Database file does not exist."""


SCHEMA_SQL = """
-- Core tables
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    file TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sections (
    id TEXT NOT NULL,
    source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    parent_id TEXT,  -- section id within same source (no FK due to composite PK)
    title TEXT NOT NULL,
    path TEXT NOT NULL,
    content TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    PRIMARY KEY (source_id, id)
);

-- FTS5 virtual table for full-text search
-- Includes source_id to correctly handle sections with same id across sources
CREATE VIRTUAL TABLE IF NOT EXISTS sections_fts USING fts5(
    source_id,
    section_id,
    title,
    content
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS sections_ai AFTER INSERT ON sections BEGIN
    INSERT INTO sections_fts(source_id, section_id, title, content)
    VALUES (NEW.source_id, NEW.id, NEW.title, NEW.content);
END;

CREATE TRIGGER IF NOT EXISTS sections_ad AFTER DELETE ON sections BEGIN
    DELETE FROM sections_fts
    WHERE source_id = OLD.source_id AND section_id = OLD.id;
END;

CREATE TRIGGER IF NOT EXISTS sections_au AFTER UPDATE ON sections BEGIN
    DELETE FROM sections_fts
    WHERE source_id = OLD.source_id AND section_id = OLD.id;
    INSERT INTO sections_fts(source_id, section_id, title, content)
    VALUES (NEW.source_id, NEW.id, NEW.title, NEW.content);
END;

-- Indexes for faster lookups
CREATE INDEX IF NOT EXISTS idx_sections_parent ON sections(source_id, parent_id);
"""


def create_schema(conn: sqlite3.Connection) -> None:
    """Create database schema with FTS5 and triggers."""
    conn.executescript(SCHEMA_SQL)


@contextmanager
def open_db(
    db_path: Path = DEFAULT_DB_PATH,
    create: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Open database connection with proper configuration.

    Args:
        db_path: Path to database file
        create: If True, create DB and schema if not exists

    Raises:
        DatabaseNotFoundError: If DB doesn't exist and create=False

    Yields:
        Configured SQLite connection
    """
    if not create and not db_path.exists():
        raise DatabaseNotFoundError(f"Database not found at {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")

    try:
        if create:
            create_schema(conn)
        yield conn
    finally:
        conn.close()


def delete_source(conn: sqlite3.Connection, source_id: str) -> None:
    """Delete a source and all its sections.

    The FTS entries are automatically cleaned up by the DELETE trigger.
    """
    # Delete sections first (triggers will clean up FTS)
    conn.execute("DELETE FROM sections WHERE source_id = ?", (source_id,))
    conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))


def index_chunks(conn: sqlite3.Connection, chunks: Iterable[Chunk]) -> int:
    """Index chunks into database, replacing existing source data.

    Args:
        conn: Database connection
        chunks: Iterable of Chunk objects to index

    Returns:
        Number of chunks indexed
    """
    chunks_list = list(chunks)
    if not chunks_list:
        return 0

    # Extract source from first chunk
    source = chunks_list[0].source

    # Delete existing data for this source (for re-indexing)
    delete_source(conn, source.id)

    # Insert source record
    conn.execute(
        "INSERT INTO sources (id, name, file) VALUES (?, ?, ?)",
        (source.id, source.name, source.file),
    )

    # Batch insert sections
    conn.executemany(
        """
        INSERT INTO sections
        (id, source_id, parent_id, title, path, content, start_line, end_line)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                chunk.section_id,
                chunk.source.id,
                chunk.parent_id,
                chunk.title,
                chunk.path,
                chunk.content,
                chunk.source.lines[0],
                chunk.source.lines[1],
            )
            for chunk in chunks_list
        ],
    )

    conn.commit()
    return len(chunks_list)


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    source_id: str | None = None,
) -> list[SearchResult]:
    """Search sections using BM25 ranking.

    Args:
        conn: Database connection
        query: Search query (FTS5 syntax supported)
        limit: Maximum results to return
        source_id: Optional filter by source

    Returns:
        List of SearchResult ordered by BM25 score (descending)
    """
    from docsearch.models import SearchResult, Source  # noqa: F811

    # BM25 returns negative scores (lower is better)
    # We negate for intuitive positive scores
    if source_id:
        cursor = conn.execute(
            """
            SELECT
                s.id,
                s.source_id,
                s.parent_id,
                s.title,
                s.path,
                s.content,
                s.start_line,
                s.end_line,
                src.name as source_name,
                src.file as source_file,
                -bm25(sections_fts) as score
            FROM sections_fts fts
            JOIN sections s ON fts.source_id = s.source_id AND fts.section_id = s.id
            JOIN sources src ON s.source_id = src.id
            WHERE sections_fts MATCH ?
              AND s.source_id = ?
            ORDER BY bm25(sections_fts)
            LIMIT ?
            """,
            (query, source_id, limit),
        )
    else:
        cursor = conn.execute(
            """
            SELECT
                s.id,
                s.source_id,
                s.parent_id,
                s.title,
                s.path,
                s.content,
                s.start_line,
                s.end_line,
                src.name as source_name,
                src.file as source_file,
                -bm25(sections_fts) as score
            FROM sections_fts fts
            JOIN sections s ON fts.source_id = s.source_id AND fts.section_id = s.id
            JOIN sources src ON s.source_id = src.id
            WHERE sections_fts MATCH ?
            ORDER BY bm25(sections_fts)
            LIMIT ?
            """,
            (query, limit),
        )

    results: list[SearchResult] = []
    for row in cursor:
        source = Source(
            id=row["source_id"],
            name=row["source_name"],
            file=row["source_file"],
            lines=(row["start_line"], row["end_line"]),
        )
        result = SearchResult(
            section_id=row["id"],
            parent_id=row["parent_id"],
            title=row["title"],
            path=row["path"],
            content=row["content"],
            source=source,
            score=row["score"],
        )
        results.append(result)

    return results
