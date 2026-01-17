import warnings

# Suppress Pydantic V1 deprecation warning from LangChain on Python 3.14+
warnings.filterwarnings("ignore", message="Core Pydantic V1 functionality")

from docsearch.cli import cli


def main() -> None:
    cli()
