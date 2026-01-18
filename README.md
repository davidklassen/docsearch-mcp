# DocSearch

Semantic search over documentation via MCP.

## Setup

```bash
uv sync
```

## Install as tool

```bash
uv cache clean && uv tool install --force .
```

## Usage

```bash
# Index a markdown file
uv run docsearch index docs/manual.md

# Search
uv run docsearch search "exception vectors"

# Pipeline mode
uv run docsearch chunks *.md | uv run docsearch index --db ./myindex.db
```

## Development

```bash
uv run ruff format src/ tests/
uv run ruff check src/ tests/
uv run mypy src/ tests/
uv run pytest
```
