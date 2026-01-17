# Claude Code Instructions

## After Code Changes

After every task that modifies code, run:

```bash
uv run ruff format src/ tests/
uv run ruff check src/ tests/
uv run mypy src/ tests/
uv run pytest
```
