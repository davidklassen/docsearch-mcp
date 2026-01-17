import json
from pathlib import Path

import click

from docsearch.chunker import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_CHUNK_SIZE,
    ChunkingStrategy,
    chunk_file,
)


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
