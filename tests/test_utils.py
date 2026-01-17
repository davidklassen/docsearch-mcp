from docsearch.utils import slugify


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
