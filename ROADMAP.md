# Roadmap

## 1. Chunker

CLI that splits a markdown file into chunks by document structure.

```bash
docsearch chunks <file.md> [--max-chunk-size <tokens>]
```

Outputs JSONL to stdout:

```json
{
  "section_id": "d1-2-3-exception-vectors",
  "parent_id": "d1-2-exception-handling",
  "title": "D1.2.3 Exception vectors",
  "path": "D1 Exception Model → D1.2 Exception handling → D1.2.3 Exception vectors",
  "content": "The vector base address must be aligned to 2KB...",
  "source": {
    "id": "arm-architecture-reference-manual",
    "name": "ARM Architecture Reference Manual",
    "file": "arm_manual.md",
    "lines": [450, 500]
  }
}
```

- `source.name`: document title from first H1 header
- `source.id`: slugified `source.name`
- `section_id`: slug from file path + header hierarchy
- `parent_id`: section_id of parent section (null for top-level)
- `path`: breadcrumb of ancestor headers
- Splits on `#`, `##`, `###` headers
- Splits sections exceeding `--max-chunk-size` (default: 2000 tokens)
- Split sections get suffix: `section-id-1`, `section-id-2`, etc. All parts share the same `parent_id`.

## 2. BM25 Index

Store chunks in SQLite with FTS5 for keyword search.

```bash
docsearch index <file.md> [--db <path>]
```

Chunks the file internally (calls milestone 1 logic), then stores. Can also read JSONL from stdin:

```bash
docsearch chunks *.md | docsearch index --db ./myindex.db
```

Default DB: `./docsearch.db`

Schema:

```sql
CREATE TABLE sources (
  id TEXT PRIMARY KEY,        -- slugified name
  name TEXT NOT NULL,         -- document title
  file TEXT NOT NULL          -- original file path
);

CREATE TABLE sections (
  id TEXT PRIMARY KEY,        -- section_id from chunk
  source_id TEXT NOT NULL REFERENCES sources(id),
  parent_id TEXT REFERENCES sections(id),  -- NULL for root sections
  title TEXT NOT NULL,
  path TEXT NOT NULL,         -- breadcrumb string (denormalized for display)
  content TEXT NOT NULL,
  start_line INTEGER NOT NULL,
  end_line INTEGER NOT NULL
);

CREATE VIRTUAL TABLE sections_fts USING fts5(
  section_id, title, content
);
```

Test CLI:

```bash
docsearch search <query> [--db <path>] [--limit <n>]
```

Outputs JSONL matching chunk format, with `score` (BM25 rank) added.

## 3. MCP Server

```bash
docsearch serve [--db <path>]
```

Starts MCP server over stdio. Configure in Claude Code's MCP settings.

Tool: `search(query: string, source_id?: string, limit?: number)`

- `source_id`: optional filter to search within one document
- `limit`: max results (default: 5)

Response:

```json
{
  "results": [
    {
      "section_id": "...",
      "parent_id": "...",
      "title": "...",
      "path": "...",
      "content": "...",
      "source": { "id": "...", "name": "...", "file": "...", "lines": [10, 25] },
      "score": 12.5
    }
  ]
}
```

## 4. Vector Search + Hybrid

Add semantic search using local embeddings.

```bash
docsearch index <file.md> --embed [--db <path>]
```

New dependencies: `sqlite-vec`, `sentence-transformers`

Embedding model: EmbeddingGemma (300M params, runs on CPU). 768-dim vectors stored in sqlite-vec.

Schema addition:

```sql
CREATE VIRTUAL TABLE section_embeddings USING vec0(
  embedding float[768],
  +section_id TEXT
);
```

Search now combines BM25 and vector results using Reciprocal Rank Fusion:

```
RRF_score(d) = 1/(k + rank_bm25(d)) + 1/(k + rank_vec(d))
```

where k=60. Results sorted by combined RRF score.

## 5. Reference Parsing

Extract cross-references at index time.

Patterns detected:
- `See section X.Y.Z`, `refer to X.Y.Z`
- Markdown links: `[text](path#anchor)`
- Inline refs: `(X.Y.Z)`

Schema addition:

```sql
CREATE TABLE references (
  from_section_id TEXT NOT NULL REFERENCES sections(id),
  to_section_id TEXT,           -- NULL if unresolved
  ref_text TEXT NOT NULL,       -- original reference text
  summary TEXT,                 -- brief summary of target section
  PRIMARY KEY (from_section_id, ref_text)
);
```

Unresolved references stored with `to_section_id = NULL`.

Summaries generated at index time using Claude (optional, requires `ANTHROPIC_API_KEY`). Falls back to target section's title if no API key.

After this milestone, search results include `references` array for each chunk.

## 6. get_section()

Direct section lookup for follow-up navigation.

Tool: `get_section(source_id: string, section_id: string)`

Response:

```json
{
  "section": { /* full chunk */ },
  "parent": { "section_id": "...", "title": "..." },
  "siblings": [
    { "section_id": "...", "title": "..." }
  ],
  "references": [
    { "section_id": "...", "title": "...", "ref_text": "See section D1.2.1" }
  ]
}
```

Returns 404-style error if section not found.

## 7. LLM Reranking

Filter retrieval candidates using Claude for relevance.

Pipeline:
1. Hybrid search returns top 50 candidates
2. Reference expansion: fetch 1-hop referenced sections, add to candidate pool
3. Send candidates to Claude Sonnet for relevance filtering
4. Return top 5

Requires `ANTHROPIC_API_KEY` env var.

Prompt structure:

```
Query: {query}

Candidates:
[1] {title} - {content_preview}
[2] ...

Select chunks that directly answer the query. Return IDs in relevance order.
```

New flag to disable (for cost/latency):

```bash
docsearch serve --no-rerank
```

## 8. Fine-tuning

Train custom embeddings for the documentation domain.

**Generate training data:**

For each chunk, use Claude to generate: "What question would this chunk answer?"

Hard negatives: chunks from same parent section (topically similar but wrong).

Target: 5-10k triplets.

**Training:**

```bash
docsearch train --triplets <triplets.jsonl> --output <model_dir>
```

Uses sentence-transformers with MultipleNegativesRankingLoss.

**Re-index:**

```bash
docsearch index --embed --model <model_dir> ...
```

**Evaluation:**

Sample 100 queries, compare retrieval recall before/after fine-tuning.
