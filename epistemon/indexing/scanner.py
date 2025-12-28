"""Directory scanning for markdown files."""

from pathlib import Path


def scan_markdown_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.md"))
