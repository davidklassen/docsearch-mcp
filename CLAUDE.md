# Claude Code Instructions

## Adding Dependencies

Use `uv add` to install new packages (ensures latest version):
```bash
uv add <package-name>
```

## After Code Changes

After every task that modifies code, run:

```bash
uv run ruff format src/ tests/
uv run ruff check src/ tests/
uv run mypy src/ tests/
uv run pytest
```
