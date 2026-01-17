import re
import unicodedata


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
