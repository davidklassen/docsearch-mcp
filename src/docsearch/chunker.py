import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from docsearch.models import Chunk, Source
from docsearch.utils import slugify

DEFAULT_MAX_CHUNK_SIZE = 2000
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

ChunkingStrategy = Literal["header", "semantic"]

# Headers to split on (level, metadata key)
HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


def extract_document_title(content: str) -> str | None:
    """Extract document title from first H1 header."""
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def build_section_id(metadata: dict[str, str]) -> str:
    """Build section ID from header metadata.

    Joins all header levels with '--' separator.
    Example: {"h1": "Doc", "h2": "Section"} -> "doc--section"
    """
    parts = []
    for key in ["h1", "h2", "h3"]:
        if key in metadata:
            parts.append(slugify(metadata[key]))
    return "--".join(parts) if parts else "root"


def build_parent_id(metadata: dict[str, str]) -> str | None:
    """Build parent section ID from header metadata.

    Returns the section_id of the parent (one level up).
    """
    parts = []
    for key in ["h1", "h2", "h3"]:
        if key in metadata:
            parts.append(slugify(metadata[key]))

    if len(parts) <= 1:
        return None  # Root level, no parent

    # Parent is all but the last part
    return "--".join(parts[:-1])


def build_path(metadata: dict[str, str]) -> str:
    """Build breadcrumb path from header metadata.

    Example: {"h1": "Doc", "h2": "Section"} -> "Doc → Section"
    """
    parts = []
    for key in ["h1", "h2", "h3"]:
        if key in metadata:
            parts.append(metadata[key])
    return " → ".join(parts) if parts else ""


def get_title(metadata: dict[str, str]) -> str:
    """Get the most specific (deepest) header as title."""
    for key in ["h3", "h2", "h1"]:
        if key in metadata:
            return metadata[key]
    return "Untitled"


def chunk_file_by_headers(
    file_path: Path, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
) -> Iterator[Chunk]:
    """Parse markdown file and yield chunks using header-based splitting."""
    content = file_path.read_text(encoding="utf-8")

    # Extract document title
    doc_title = extract_document_title(content)
    if doc_title is None:
        doc_title = file_path.stem

    source_id = slugify(doc_title)

    # Split by headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    header_splits = header_splitter.split_text(content)

    # Further split large chunks by size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size * 4,  # Approximate chars from tokens
        chunk_overlap=0,
        length_function=len,
    )

    # Track seen section_ids to handle duplicates from size splits
    seen_ids: dict[str, int] = {}

    for doc in header_splits:
        metadata = doc.metadata
        base_section_id = build_section_id(metadata)
        parent_id = build_parent_id(metadata)
        path = build_path(metadata)
        title = get_title(metadata)

        # Split if content is too large
        if len(doc.page_content) > max_chunk_size * 4:
            sub_docs = text_splitter.split_text(doc.page_content)
        else:
            sub_docs = [doc.page_content]

        for sub_content in sub_docs:
            # Handle duplicate section_ids (from size splitting)
            if base_section_id in seen_ids:
                seen_ids[base_section_id] += 1
                section_id = f"{base_section_id}-{seen_ids[base_section_id]}"
            else:
                seen_ids[base_section_id] = 0
                section_id = base_section_id

            yield Chunk(
                section_id=section_id,
                parent_id=parent_id,
                title=title,
                path=path,
                content=sub_content.strip(),
                source=Source(
                    id=source_id,
                    name=doc_title,
                    file=str(file_path),
                    lines=(0, 0),  # LangChain doesn't track line numbers
                ),
            )


def chunk_file_semantic(
    file_path: Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    breakpoint_threshold: float = 95.0,
) -> Iterator[Chunk]:
    """Parse markdown file and yield chunks using semantic splitting."""
    # Lazy import to avoid loading models when not needed
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_huggingface import HuggingFaceEmbeddings

    content = file_path.read_text(encoding="utf-8")

    # Extract document title
    doc_title = extract_document_title(content)
    if doc_title is None:
        doc_title = file_path.stem

    source_id = slugify(doc_title)

    # Initialize embeddings
    encode_kwargs: dict[str, str] = {}
    if "embeddinggemma" in model_name.lower():
        encode_kwargs["prompt_name"] = "document"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs=encode_kwargs,
    )

    # Create semantic chunker
    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_threshold,
    )

    # Split text
    docs = chunker.create_documents([content])

    for i, doc in enumerate(docs):
        section_id = f"{source_id}--chunk-{i + 1}"

        yield Chunk(
            section_id=section_id,
            parent_id=None,  # Semantic chunks don't have hierarchy
            title=f"Chunk {i + 1}",
            path=doc_title,
            content=doc.page_content.strip(),
            source=Source(
                id=source_id,
                name=doc_title,
                file=str(file_path),
                lines=(0, 0),
            ),
        )


def chunk_file(
    file_path: Path,
    strategy: ChunkingStrategy = "header",
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    breakpoint_threshold: float = 95.0,
) -> Iterator[Chunk]:
    """Parse markdown file and yield chunks.

    Args:
        file_path: Path to markdown file
        strategy: "header" for structure-based, "semantic" for embedding-based
        max_chunk_size: Max tokens per chunk (header strategy only)
        model_name: Embedding model for semantic chunking
        breakpoint_threshold: Percentile threshold for semantic splits (0-100)
    """
    if strategy == "semantic":
        yield from chunk_file_semantic(file_path, model_name, breakpoint_threshold)
    else:
        yield from chunk_file_by_headers(file_path, max_chunk_size)
