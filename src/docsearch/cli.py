import json
import sys
from pathlib import Path

import click

from docsearch.chunker import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_CHUNK_SIZE,
    ChunkingStrategy,
    chunk_file,
)
from docsearch.db import DEFAULT_DB_PATH, DatabaseNotFoundError, index_chunks, open_db
from docsearch.db import search as db_search
from docsearch.models import Chunk


@click.group()
def cli() -> None:
    """DocSearch - Semantic search over technical documentation."""


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--strategy",
    type=click.Choice(["header", "semantic"]),
    default="header",
    help="Chunking strategy: header (structure-based) or semantic (embedding-based)",
)
@click.option(
    "--max-chunk-size",
    default=DEFAULT_MAX_CHUNK_SIZE,
    help=f"Max tokens per chunk, header only (default: {DEFAULT_MAX_CHUNK_SIZE})",
)
@click.option(
    "--model",
    default=DEFAULT_EMBEDDING_MODEL,
    help=f"Embedding model for semantic chunking (default: {DEFAULT_EMBEDDING_MODEL})",
)
@click.option(
    "--threshold",
    default=95.0,
    help="Percentile threshold for semantic splits, 0-100 (default: 95.0)",
)
def chunks(
    file: Path,
    strategy: ChunkingStrategy,
    max_chunk_size: int,
    model: str,
    threshold: float,
) -> None:
    """Split markdown file into chunks, output JSONL to stdout."""
    for chunk in chunk_file(
        file,
        strategy=strategy,
        max_chunk_size=max_chunk_size,
        model_name=model,
        breakpoint_threshold=threshold,
    ):
        click.echo(json.dumps(chunk.to_dict()))


@cli.command()
@click.argument("file", type=click.Path(exists=True, path_type=Path), required=False)
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    default=DEFAULT_DB_PATH,
    help=f"Database path (default: {DEFAULT_DB_PATH})",
)
@click.option(
    "--strategy",
    type=click.Choice(["header", "semantic"]),
    default="header",
    help="Chunking strategy: header (structure-based) or semantic (embedding-based)",
)
def index(
    file: Path | None,
    db: Path,
    strategy: ChunkingStrategy,
) -> None:
    """Index markdown file(s) into the search database.

    If FILE is provided, chunks and indexes that file.
    If stdin is piped, reads JSONL chunks from stdin.

    Examples:

        docsearch index docs/manual.md

        docsearch chunks *.md | docsearch index --db ./myindex.db
    """
    chunks_to_index: list[Chunk] = []

    if file is not None:
        # File mode: chunk the file and index
        chunks_to_index = list(chunk_file(file, strategy=strategy))
    elif not sys.stdin.isatty():
        # Stdin mode: read JSONL from stdin
        for line in sys.stdin:
            line = line.strip()
            if line:
                data = json.loads(line)
                chunks_to_index.append(Chunk.from_dict(data))
    else:
        raise click.UsageError("Either provide a FILE argument or pipe JSONL to stdin.")

    if not chunks_to_index:
        click.echo("No chunks to index.", err=True)
        return

    with open_db(db, create=True) as conn:
        count = index_chunks(conn, chunks_to_index)

    source_name = chunks_to_index[0].source.name
    click.echo(f"Indexed {count} chunks from {source_name}")


@cli.command()
@click.argument("query")
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    default=DEFAULT_DB_PATH,
    help=f"Database path (default: {DEFAULT_DB_PATH})",
)
@click.option(
    "--limit",
    "-n",
    type=int,
    default=10,
    help="Maximum results (default: 10)",
)
@click.option(
    "--source",
    "-s",
    type=str,
    default=None,
    help="Filter by source ID",
)
def search(
    query: str,
    db: Path,
    limit: int,
    source: str | None,
) -> None:
    """Search indexed documentation.

    Outputs JSONL with matching chunks and BM25 scores.

    Example:

        docsearch search "exception vectors" --limit 5
    """
    try:
        with open_db(db, create=False) as conn:
            results = db_search(conn, query, limit=limit, source_id=source)
    except DatabaseNotFoundError as err:
        raise click.ClickException(
            f"Database not found at {db}. Run 'docsearch index' first."
        ) from err

    for result in results:
        click.echo(json.dumps(result.to_dict()))


def expand_globs(patterns: tuple[str, ...]) -> list[Path]:
    """Expand glob patterns to file paths."""
    files: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        # If it's an existing file, use it directly
        if path.exists() and path.is_file():
            files.append(path)
        # Otherwise treat as glob pattern
        elif "*" in pattern or "?" in pattern:
            # Handle absolute vs relative patterns
            if path.is_absolute():
                # For absolute paths, glob from root
                matches = list(Path("/").glob(pattern.lstrip("/")))
            else:
                matches = list(Path.cwd().glob(pattern))
            files.extend(sorted(matches))
        else:
            # Not a glob, not an existing file - will error later
            files.append(path)
    return files


@cli.command()
@click.argument("patterns", nargs=-1, required=True)
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    default=None,
    help="Database path (default: temp file, deleted on exit)",
)
def serve(patterns: tuple[str, ...], db: Path | None) -> None:
    """Start the MCP server over stdio, auto-indexing the specified files.

    Supports glob patterns (e.g., docs/**/*.md for recursive matching).

    Examples:

        docsearch serve docs/*.md

        docsearch serve "docs/**/*.md" --db ./docsearch.db
    """
    from docsearch.server import init_server, run_server

    files = expand_globs(patterns)

    if not files:
        raise click.ClickException(f"No files found matching: {' '.join(patterns)}")

    # Validate all files exist
    for f in files:
        if not f.exists():
            raise click.ClickException(f"File not found: {f}")
        if not f.is_file():
            raise click.ClickException(f"Not a file: {f}")

    init_server(files, db)
    run_server()
