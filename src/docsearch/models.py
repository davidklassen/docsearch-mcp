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
