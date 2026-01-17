"""MCP server for docsearch."""

import logging
import tempfile
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from docsearch.chunker import chunk_file
from docsearch.db import index_chunks, open_db
from docsearch.db import search as db_search

# Configure logging to stderr (CRITICAL: never use print/stdout in MCP servers)
logging.basicConfig(
    level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Module-level state (set by serve command before run)
_db_path: Path | None = None

mcp = FastMCP("docsearch")


def init_server(files: list[Path], db_path: Path | None = None) -> None:
    """Initialize server: index files and set DB path."""
    global _db_path

    # Use temp file if no db_path specified
    if db_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        _db_path = Path(tmp.name)
    else:
        _db_path = db_path

    # Index all files
    with open_db(_db_path, create=True) as conn:
        for file_path in files:
            chunks = list(chunk_file(file_path))
            if chunks:
                count = index_chunks(conn, chunks)
                logger.info(f"Indexed {count} chunks from {file_path.name}")


@mcp.tool()
def search(query: str, source_id: str | None = None, limit: int = 5) -> dict[str, Any]:
    """Search indexed documentation using BM25 full-text search.

    Args:
        query: Search query (FTS5 syntax supported)
        source_id: Optional filter to search within one document
        limit: Maximum results to return (default: 5)

    Returns:
        Dictionary with "results" array containing matching chunks with scores
    """
    if _db_path is None:
        return {"error": "Server not initialized"}

    with open_db(_db_path, create=False) as conn:
        results = db_search(conn, query, limit=limit, source_id=source_id)
    return {"results": [r.to_dict() for r in results]}


def run_server() -> None:
    """Run the MCP server with stdio transport."""
    mcp.run(transport="stdio")
