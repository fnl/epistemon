"""Directory scanning for markdown files."""

from pathlib import Path


def scan_markdown_files(directory: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*.md" if recursive else "*.md"
    return sorted(directory.glob(pattern))
