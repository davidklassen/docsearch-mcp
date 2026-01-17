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

from docsearch.embedding import EMBEDDING_DIM

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
    from docsearch.utils import escape_fts5_query

    # Escape query for safe FTS5 matching
    escaped_query = escape_fts5_query(query)
    if not escaped_query:
        return []

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
            (escaped_query, source_id, limit),
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
            (escaped_query, limit),
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


# --- Embedding / Vector Search Support ---


def load_vec_extension(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec extension into connection."""
    import sqlite_vec  # type: ignore[import-untyped]

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def create_embedding_schema(conn: sqlite3.Connection) -> None:
    """Create vec0 virtual table for embeddings."""
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS section_embeddings USING vec0(
            embedding float[{EMBEDDING_DIM}],
            source_id TEXT,
            section_id TEXT
        )
    """)
    conn.commit()


def store_embeddings(
    conn: sqlite3.Connection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> int:
    """Store embeddings for chunks.

    Args:
        conn: Database connection (with vec extension loaded)
        chunks: List of Chunk objects
        embeddings: Corresponding embedding vectors

    Returns:
        Number of embeddings stored
    """
    import sqlite_vec

    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have same length")

    if not chunks:
        return 0

    for chunk, embedding in zip(chunks, embeddings, strict=True):
        conn.execute(
            """
            INSERT INTO section_embeddings (embedding, source_id, section_id)
            VALUES (?, ?, ?)
            """,
            (
                sqlite_vec.serialize_float32(embedding),
                chunk.source.id,
                chunk.section_id,
            ),
        )

    conn.commit()
    return len(chunks)


def delete_source_embeddings(conn: sqlite3.Connection, source_id: str) -> None:
    """Delete embeddings for a source (for re-indexing)."""
    conn.execute(
        "DELETE FROM section_embeddings WHERE source_id = ?",
        (source_id,),
    )
    conn.commit()


def search_vectors(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    limit: int = 10,
    source_id: str | None = None,
) -> list[tuple[str, str, float]]:
    """Search by vector similarity using KNN.

    Args:
        conn: Database connection (with vec extension loaded)
        query_embedding: Query vector
        limit: Maximum results to return
        source_id: Optional filter by source

    Returns:
        List of (source_id, section_id, distance) tuples,
        ordered by distance (ascending)
    """
    import sqlite_vec

    query_blob = sqlite_vec.serialize_float32(query_embedding)

    if source_id:
        cursor = conn.execute(
            """
            SELECT source_id, section_id, distance
            FROM section_embeddings
            WHERE embedding MATCH ? AND k = ? AND source_id = ?
            ORDER BY distance
            """,
            (query_blob, limit, source_id),
        )
    else:
        cursor = conn.execute(
            """
            SELECT source_id, section_id, distance
            FROM section_embeddings
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (query_blob, limit),
        )

    return [(row[0], row[1], row[2]) for row in cursor]


def rrf_score(rank: int, k: int = 60) -> float:
    """Compute single RRF score component: 1/(k + rank).

    Args:
        rank: 1-based rank of result
        k: RRF constant (default 60)

    Returns:
        RRF score component
    """
    return 1.0 / (k + rank)


def compute_rrf_scores(
    bm25_results: list[tuple[str, str]],  # (source_id, section_id)
    vec_results: list[tuple[str, str]],
    k: int = 60,
) -> dict[tuple[str, str], float]:
    """Combine BM25 and vector ranks into RRF scores.

    Args:
        bm25_results: List of (source_id, section_id) from BM25 search
        vec_results: List of (source_id, section_id) from vector search
        k: RRF constant

    Returns:
        Dict mapping (source_id, section_id) to combined RRF score
    """
    scores: dict[tuple[str, str], float] = {}

    # Add BM25 contributions (rank is 1-based)
    for rank, key in enumerate(bm25_results, start=1):
        scores[key] = scores.get(key, 0.0) + rrf_score(rank, k)

    # Add vector contributions (rank is 1-based)
    for rank, key in enumerate(vec_results, start=1):
        scores[key] = scores.get(key, 0.0) + rrf_score(rank, k)

    return scores


def _fetch_section(
    conn: sqlite3.Connection,
    source_id: str,
    section_id: str,
) -> SearchResult | None:
    """Fetch a section by ID and return as SearchResult (with score=0)."""
    from docsearch.models import SearchResult, Source

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
            src.file as source_file
        FROM sections s
        JOIN sources src ON s.source_id = src.id
        WHERE s.source_id = ? AND s.id = ?
        """,
        (source_id, section_id),
    )
    row = cursor.fetchone()
    if row is None:
        return None

    source = Source(
        id=row["source_id"],
        name=row["source_name"],
        file=row["source_file"],
        lines=(row["start_line"], row["end_line"]),
    )
    return SearchResult(
        section_id=row["id"],
        parent_id=row["parent_id"],
        title=row["title"],
        path=row["path"],
        content=row["content"],
        source=source,
        score=0.0,  # Will be set by caller
    )


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    query_embedding: list[float],
    limit: int = 10,
    source_id: str | None = None,
    k: int = 60,
) -> list[SearchResult]:
    """Hybrid BM25 + vector search with RRF.

    Combines BM25 full-text search and vector similarity search using
    Reciprocal Rank Fusion (RRF) to produce ranked results.

    Args:
        conn: Database connection
        query: Search query text
        query_embedding: Query embedding vector
        limit: Maximum results to return
        source_id: Optional filter by source
        k: RRF constant

    Returns:
        List of SearchResult ordered by RRF score (descending)
    """
    from docsearch.models import SearchResult  # noqa: F811

    # Get more candidates than needed for RRF fusion
    candidate_limit = limit * 2

    # Run BM25 search
    bm25_results = search(conn, query, limit=candidate_limit, source_id=source_id)
    bm25_keys = [(r.source.id, r.section_id) for r in bm25_results]

    # Run vector search
    vec_raw = search_vectors(
        conn, query_embedding, limit=candidate_limit, source_id=source_id
    )
    vec_keys = [(src_id, sec_id) for src_id, sec_id, _ in vec_raw]

    # Compute RRF scores
    rrf_scores = compute_rrf_scores(bm25_keys, vec_keys, k=k)

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Fetch section data and build results
    results: list[SearchResult] = []
    for src_id, sec_id in sorted_keys[:limit]:
        result = _fetch_section(conn, src_id, sec_id)
        if result is not None:
            # Update score with RRF score
            result = SearchResult(
                section_id=result.section_id,
                parent_id=result.parent_id,
                title=result.title,
                path=result.path,
                content=result.content,
                source=result.source,
                score=rrf_scores[(src_id, sec_id)],
            )
            results.append(result)

    return results
