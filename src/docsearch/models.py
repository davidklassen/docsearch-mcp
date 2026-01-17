from dataclasses import dataclass
from typing import Any


@dataclass
class Source:
    """Metadata about the source document."""

    id: str  # slugified name
    name: str  # document title (first H1)
    file: str  # original file path
    lines: tuple[int, int]  # (start_line, end_line), 1-indexed

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "file": self.file,
            "lines": list(self.lines),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Source":
        """Reconstruct Source from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            file=data["file"],
            lines=(data["lines"][0], data["lines"][1]),
        )


@dataclass
class Chunk:
    """A chunk of documentation content."""

    section_id: str
    parent_id: str | None
    title: str
    path: str
    content: str
    source: Source

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "parent_id": self.parent_id,
            "title": self.title,
            "path": self.path,
            "content": self.content,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        """Reconstruct Chunk from dictionary."""
        return cls(
            section_id=data["section_id"],
            parent_id=data["parent_id"],
            title=data["title"],
            path=data["path"],
            content=data["content"],
            source=Source.from_dict(data["source"]),
        )


@dataclass
class SearchResult:
    """A search result with relevance score."""

    section_id: str
    parent_id: str | None
    title: str
    path: str
    content: str
    source: Source
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "parent_id": self.parent_id,
            "title": self.title,
            "path": self.path,
            "content": self.content,
            "source": self.source.to_dict(),
            "score": self.score,
        }

    @classmethod
    def from_chunk(cls, chunk: Chunk, score: float) -> "SearchResult":
        """Create SearchResult from Chunk with score."""
        return cls(
            section_id=chunk.section_id,
            parent_id=chunk.parent_id,
            title=chunk.title,
            path=chunk.path,
            content=chunk.content,
            source=chunk.source,
            score=score,
        )
