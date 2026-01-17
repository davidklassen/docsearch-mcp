from docsearch.utils import escape_fts5_query, slugify


def test_escape_fts5_query_hyphenated_term() -> None:
    """Hyphenated terms are quoted to prevent FTS5 NOT operator."""
    assert escape_fts5_query("callee-saved") == '"callee-saved"'


def test_escape_fts5_query_simple_term() -> None:
    """Simple terms are quoted."""
    assert escape_fts5_query("vectors") == '"vectors"'


def test_escape_fts5_query_multiple_terms() -> None:
    """Multiple terms are individually quoted."""
    assert escape_fts5_query("exception vectors") == '"exception" "vectors"'


def test_escape_fts5_query_multiple_spaces() -> None:
    """Multiple spaces between terms are handled."""
    assert escape_fts5_query("hello   world") == '"hello" "world"'


def test_escape_fts5_query_empty_string() -> None:
    """Empty string returns empty string."""
    assert escape_fts5_query("") == ""


def test_escape_fts5_query_whitespace_only() -> None:
    """Whitespace-only returns empty string."""
    assert escape_fts5_query("   ") == ""


def test_escape_fts5_query_special_chars() -> None:
    """FTS5 special characters are safely quoted."""
    assert escape_fts5_query("title:test") == '"title:test"'
    assert escape_fts5_query("+required") == '"+required"'
    assert escape_fts5_query("prefix*") == '"prefix*"'
    assert escape_fts5_query("a|b") == '"a|b"'
    assert escape_fts5_query("(group)") == '"(group)"'


def test_escape_fts5_query_internal_quotes() -> None:
    """Internal double quotes are escaped."""
    # Input: 'say "hello"' splits into ['say', '"hello"']
    # '"hello"' becomes '"""hello"""' (quotes escaped and wrapped)
    assert escape_fts5_query('say "hello"') == '"say" """hello"""'


def test_escape_fts5_query_complex() -> None:
    """Complex query with multiple special chars."""
    result = escape_fts5_query("callee-saved VBAR_EL1 test:column")
    assert result == '"callee-saved" "VBAR_EL1" "test:column"'


def test_slugify_simple() -> None:
    assert slugify("Hello World") == "hello-world"


def test_slugify_with_dots() -> None:
    assert slugify("D1.2.3 Exception vectors") == "d1-2-3-exception-vectors"


def test_slugify_with_underscores() -> None:
    assert slugify("VBAR_EL1 Register") == "vbar-el1-register"


def test_slugify_full_title() -> None:
    result = slugify("ARM Architecture Reference Manual")
    assert result == "arm-architecture-reference-manual"


def test_slugify_special_chars() -> None:
    assert slugify("Hello! World?") == "hello-world"


def test_slugify_multiple_spaces() -> None:
    assert slugify("Hello    World") == "hello-world"


def test_slugify_leading_trailing() -> None:
    assert slugify("  Hello World  ") == "hello-world"


def test_slugify_empty() -> None:
    assert slugify("") == ""
