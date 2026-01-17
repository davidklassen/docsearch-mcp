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
from docsearch.db import (
    DEFAULT_DB_PATH,
    DatabaseNotFoundError,
    create_embedding_schema,
    delete_source_embeddings,
    hybrid_search,
    index_chunks,
    load_vec_extension,
    open_db,
    store_embeddings,
)
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
@click.argument("patterns", nargs=-1, required=False)
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
    patterns: tuple[str, ...],
    db: Path,
    strategy: ChunkingStrategy,
) -> None:
    """Index markdown file(s) into the search database.

    Accepts file paths or glob patterns. If stdin is piped, reads JSONL chunks.

    Examples:

        docsearch index docs/manual.md

        docsearch index docs/**/*.md --db ./myindex.db

        docsearch chunks file.md | docsearch index --db ./myindex.db
    """
    from docsearch.embedding import Embedder

    files: list[Path] = []
    chunks_from_stdin: list[Chunk] = []

    if patterns:
        # Expand glob patterns
        files = expand_globs(patterns)
        if not files:
            raise click.ClickException(f"No files found matching: {' '.join(patterns)}")
        # Validate all files exist
        for f in files:
            if not f.exists():
                raise click.ClickException(f"File not found: {f}")
            if not f.is_file():
                raise click.ClickException(f"Not a file: {f}")
    elif not sys.stdin.isatty():
        # Stdin mode: read JSONL from stdin
        for line in sys.stdin:
            line = line.strip()
            if line:
                data = json.loads(line)
                chunks_from_stdin.append(Chunk.from_dict(data))
    else:
        raise click.UsageError("Provide file patterns or pipe JSONL to stdin.")

    if not files and not chunks_from_stdin:
        click.echo("No files or chunks to index.", err=True)
        return

    embedder = Embedder()
    total_chunks = 0
    total_embeddings = 0

    with open_db(db, create=True) as conn:
        # Load vec extension and create schema
        load_vec_extension(conn)
        create_embedding_schema(conn)

        # Index files
        for file_path in files:
            chunks = list(chunk_file(file_path, strategy=strategy))
            if chunks:
                count = index_chunks(conn, chunks)

                # Delete existing embeddings for this source
                source_id = chunks[0].source.id
                delete_source_embeddings(conn, source_id)

                # Generate and store embeddings
                texts = [chunk.content for chunk in chunks]
                embeddings = embedder.embed(texts)
                embed_count = store_embeddings(conn, chunks, embeddings)

                total_chunks += count
                total_embeddings += embed_count
                click.echo(f"Indexed {count} chunks from {file_path.name}")

        # Index chunks from stdin
        if chunks_from_stdin:
            count = index_chunks(conn, chunks_from_stdin)
            source_id = chunks_from_stdin[0].source.id
            delete_source_embeddings(conn, source_id)

            texts = [chunk.content for chunk in chunks_from_stdin]
            embeddings = embedder.embed(texts)
            embed_count = store_embeddings(conn, chunks_from_stdin, embeddings)

            total_chunks += count
            total_embeddings += embed_count
            source_name = chunks_from_stdin[0].source.name
            click.echo(f"Indexed {count} chunks from {source_name}")

    click.echo(f"Total: {total_chunks} chunks, {total_embeddings} embeddings")


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
    """Search indexed documentation using hybrid BM25 + vector search.

    Outputs JSONL with matching chunks and scores.

    Example:

        docsearch search "exception vectors" --limit 5
    """
    from docsearch.embedding import Embedder

    try:
        with open_db(db, create=False) as conn:
            load_vec_extension(conn)
            embedder = Embedder()
            query_embedding = embedder.embed_single(query)
            results = hybrid_search(
                conn,
                query,
                query_embedding=query_embedding,
                limit=limit,
                source_id=source,
            )
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
@click.argument("patterns", nargs=-1, required=False)
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    default=None,
    help="Database path (default: temp file, deleted on exit)",
)
@click.option(
    "--skip-index",
    is_flag=True,
    default=False,
    help="Skip indexing, use existing database (requires --db)",
)
def serve(patterns: tuple[str, ...], db: Path | None, skip_index: bool) -> None:
    """Start the MCP server over stdio, auto-indexing the specified files.

    Supports glob patterns (e.g., docs/**/*.md for recursive matching).

    Examples:

        docsearch serve docs/*.md

        docsearch serve "docs/**/*.md" --db ./docsearch.db

        # Pre-index, then serve without re-indexing:
        docsearch index docs/*.md --db ./docsearch.db
        docsearch serve --db ./docsearch.db --skip-index
    """
    from docsearch.server import init_server, run_server

    if skip_index:
        if db is None:
            raise click.ClickException("--skip-index requires --db to specify database")
        if not db.exists():
            raise click.ClickException(f"Database not found: {db}")
        init_server([], db, skip_index=True)
        run_server()
        return

    if not patterns:
        raise click.ClickException(
            "File patterns required (or use --skip-index with --db)"
        )

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
