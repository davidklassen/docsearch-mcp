# DocSearch MCP - Design Document

## Overview

An MCP server that provides semantic search over large technical documentation (ARM architecture manuals, etc.) for Claude Code. Enables a coding agent to query 10k+ page documents without context flooding.

## Problem Statement

1. **Documents too large for context** - ARM Architecture Reference Manual is 10k+ pages; won't fit in LLM context even as text
2. **Naive retrieval floods context** - Top-K chunks often include marginally relevant content that wastes tokens and confuses the agent
3. **Technical terminology** - Terms like `VBAR_EL1`, `TTBR0_EL1` need exact matching, not just semantic similarity
4. **Cross-references everywhere** - ARM docs heavily reference other sections ("See D1.2.3"); retrieved chunks may be incomplete without referenced content
5. **OCR quality** - PDF conversion (marker) struggles with tables, register layouts, bit fields

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Claude Code (coding agent context)                         │
│                                                             │
│  MCP call: docsearch.search("exception vectors EL1")        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  DocSearch MCP Server (Python)                              │
│                                                             │
│  1. Hybrid Retrieval (parallel)                             │
│     ├── BM25 (SQLite FTS5) → exact term matches             │
│     └── Vector (sqlite-vec) → semantic matches              │
│                                                             │
│  2. Reference Expansion                                     │
│     └── Fetch 1-hop referenced chunks into candidate pool   │
│                                                             │
│  3. LLM Reranking (separate API call)                       │
│     └── Claude filters to high-relevance chunks only        │
│                                                             │
│  4. Return top 3-5 chunks with metadata                     │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Document Ingestion Pipeline

**Input**: PDF documents
**Output**: Indexed chunks in SQLite

```
PDF → marker (MD conversion) → Chunker → Indexer → SQLite
                                  │
                                  ├── Extract section hierarchy
                                  ├── Parse cross-references
                                  └── Generate inline summaries for refs
```

**Chunking strategy**:
- Use document structure (sections, subsections) as boundaries, not fixed token counts
- Preserve section IDs for reference resolution (e.g., "D1.2.3")
- Store breadcrumb path: `D1 → D1.2 → D1.2.3`
- Include inline summaries for outgoing references

**Reference parsing**:
- Regex patterns for common formats: "See section X.Y.Z", "refer to page N"
- Store as edges: `chunk_id → references → [target_ids]`
- Generate brief summaries of referenced sections at index time

### 2. Hybrid Search Index

**Storage**: Single SQLite database with two indexes

| Component | Implementation | Purpose |
|-----------|----------------|---------|
| Keyword search | SQLite FTS5 (BM25) | Exact term matching (`VBAR_EL1`) |
| Vector search | sqlite-vec | Semantic similarity |
| Metadata | Regular tables | Section hierarchy, references, source info |

**Embedding model**: EmbeddingGemma (300M params)
- Fine-tune on synthetic ARM documentation query-document pairs
- 768-dim vectors, can truncate to 256 for storage efficiency

### 3. LLM Reranker

**Purpose**: Filter 30-50 retrieval candidates down to 3-5 high-relevance chunks

**Implementation**: Separate Claude API call (disposable context)

```python
async def rerank(query: str, candidates: list[Chunk]) -> list[Chunk]:
    prompt = f"""
    Query: {query}

    Candidate chunks:
    {format_candidates(candidates)}

    Select only chunks that directly answer the query.
    If a chunk references another and that reference is essential, include both.
    Return chunk IDs in relevance order.
    """
    response = await claude.messages.create(...)
    return filter_by_ids(candidates, parse_ids(response))
```

**Why separate context**: Protects the coding agent's context from noise. The reranker can read all candidates thoroughly; the agent only sees refined results.

### 4. MCP Interface

**Tools exposed to Claude Code**:

```python
@mcp.tool()
async def search(query: str) -> list[Chunk]:
    """Search documentation. Returns relevant chunks across all indexed documents."""

@mcp.tool()
async def get_section(source_id: str, section_id: str) -> Chunk:
    """Fetch a specific section by ID (e.g., 'D1.2.3') from a specific document."""
```

Two tools only. Simple interface, agent doesn't need to understand internals.

**Chunk response format**:
```json
{
  "section_id": "D1.2.3",
  "title": "Exception vectors",
  "path": "D1 Exception Model → D1.2 Exception handling → D1.2.3",
  "content": "The vector base address must be aligned to 2KB...",
  "source": {
    "id": "arm-arm",
    "name": "ARM Architecture Reference Manual",
    "file": "docs/arm_manual.md",
    "lines": [450, 500]
  },
  "references": [
    {"id": "D1.2.1", "summary": "Exception types and their priority..."}
  ]
}
```

- `source.id` identifies the document (needed for `get_section` calls and cross-document references)
- `source.name` is the human-readable document title
- `source.file` and `source.lines` enable the agent to read additional context directly from the markdown file when needed

## Fine-tuning Pipeline

**Goal**: Improve EmbeddingGemma's understanding of ARM terminology

**Training data format**: Triplets (query, positive, hard_negative)

```python
[
  ("What register holds the exception vector base address?",
   "VBAR_EL1 contains the vector base address for exceptions taken to EL1",
   "SCTLR_EL1 controls system configuration at EL1"),  # hard negative
]
```

**Synthetic data generation**:
1. For each chunk, use LLM to generate: "What question would this chunk answer?"
2. Hard negatives: other chunks from same section (topically similar but wrong answer)
3. Target: 5k-10k triplets for initial fine-tuning

**Training**:
```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss

model = SentenceTransformer("google/embeddinggemma-300M")
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=dataset,
    loss=MultipleNegativesRankingLoss(model),
    args=TrainingArguments(learning_rate=2e-5, num_train_epochs=5)
)
trainer.train()
```

## Tech Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| MCP Server | Python + mcp library | Native integration with ML tooling |
| PDF conversion | marker | Best OSS PDF-to-markdown |
| Database | SQLite | Single file, portable, no external services |
| BM25 search | SQLite FTS5 | Built-in, fast, no dependencies |
| Vector search | sqlite-vec | Keeps everything in one DB |
| Embeddings | EmbeddingGemma | Fine-tunable, small, runs locally |
| Reranker | Claude API (Sonnet) | Best relevance judgment for technical content |
| Fine-tuning | sentence-transformers | Standard, well-documented |

## Open Questions

1. **Marker quality validation** - Need to manually inspect marker output on ARM docs, especially tables and register definitions. May need post-processing or alternative extraction for structured data.

2. **Chunk size tuning** - Start with section-based chunking, but may need to split very large sections or merge very small ones. Requires experimentation.

3. **Evaluation approach** - Use LLM-as-judge for automated relevance scoring, spot-check manually to calibrate.
