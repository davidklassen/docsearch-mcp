import re
import unicodedata


def escape_fts5_query(query: str) -> str:
    """Escape user query for safe FTS5 MATCH.

    Wraps each term in double quotes to prevent FTS5 operator interpretation.
    This ensures hyphenated terms like "callee-saved" are treated literally
    rather than as "callee NOT saved".

    Args:
        query: Raw user query string

    Returns:
        FTS5-safe query with terms quoted
    """
    if not query or not query.strip():
        return ""

    terms = query.split()
    quoted_terms = []
    for term in terms:
        if term:
            escaped = term.replace('"', '""')
            quoted_terms.append(f'"{escaped}"')

    return " ".join(quoted_terms)


def slugify(text: str) -> str:
    """Convert text to URL-safe slug.

    Examples:
        'ARM Architecture Reference Manual' -> 'arm-architecture-reference-manual'
        'D1.2.3 Exception vectors' -> 'd1-2-3-exception-vectors'
        'VBAR_EL1 Register' -> 'vbar-el1-register'
    """
    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Lowercase
    text = text.lower()

    # Replace dots, underscores, spaces with hyphens
    text = re.sub(r"[.\s_]+", "-", text)

    # Remove non-alphanumeric characters (except hyphens)
    text = re.sub(r"[^a-z0-9-]", "", text)

    # Collapse multiple hyphens
    text = re.sub(r"-+", "-", text)

    # Strip leading/trailing hyphens
    text = text.strip("-")

    return text
