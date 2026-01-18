"""MCP server for docsearch."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP

from docsearch.chunker import chunk_file
from docsearch.db import (
    create_embedding_schema,
    delete_source_embeddings,
    hybrid_search,
    index_chunks,
    load_vec_extension,
    open_db,
    store_embeddings,
)

if TYPE_CHECKING:
    from docsearch.embedding import Embedder

# Configure logging to stderr (CRITICAL: never use print/stdout in MCP servers)
logging.basicConfig(
    level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Module-level state (set by serve command before run)
_db_path: Path | None = None
_embedder: Embedder | None = None

mcp = FastMCP("docsearch")


def init_server(
    files: list[Path], db_path: Path | None = None, skip_index: bool = False
) -> None:
    """Initialize server: index files and set DB path.

    Args:
        files: List of markdown files to index
        db_path: Optional database path (default: temp file)
        skip_index: If True, skip indexing and use existing database
    """
    global _db_path, _embedder

    from docsearch.embedding import Embedder

    # Use temp file if no db_path specified
    if db_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        _db_path = Path(tmp.name)
    else:
        _db_path = db_path

    # Initialize embedder
    _embedder = Embedder()

    if skip_index:
        # Just verify DB exists and has embedding schema
        with open_db(_db_path, create=False) as conn:
            load_vec_extension(conn)
        logger.info(f"Using existing database: {_db_path}")
        return

    # Index all files
    with open_db(_db_path, create=True) as conn:
        load_vec_extension(conn)
        create_embedding_schema(conn)

        for file_path in files:
            chunks = list(chunk_file(file_path))
            if chunks:
                count = index_chunks(conn, chunks)

                # Delete existing embeddings for this source
                source_id = chunks[0].source.id
                delete_source_embeddings(conn, source_id)

                # Generate and store embeddings
                texts = [chunk.content for chunk in chunks]
                embeddings = _embedder.embed(texts)
                embed_count = store_embeddings(conn, chunks, embeddings)
                logger.info(
                    f"Indexed {count} chunks from {file_path.name} "
                    f"({embed_count} embeddings)"
                )


@mcp.tool()
def search(query: str) -> dict[str, Any]:
    """Search indexed documentation using hybrid BM25 + vector search.

    Args:
        query: Search query (FTS5 syntax supported for BM25)

    Returns:
        Dictionary with "results" array containing matching chunks with scores
    """
    if _db_path is None or _embedder is None:
        return {"error": "Server not initialized"}

    with open_db(_db_path, create=False) as conn:
        load_vec_extension(conn)
        query_embedding = _embedder.embed_single(query)
        results = hybrid_search(conn, query, query_embedding=query_embedding, limit=6)

    return {"results": [r.to_dict() for r in results]}


def run_server() -> None:
    """Run the MCP server with stdio transport."""
    mcp.run(transport="stdio")
