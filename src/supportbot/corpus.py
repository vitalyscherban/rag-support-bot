"""Corpus loading."""

from __future__ import annotations

from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "corpus"


def load_corpus(directory: str | Path = DEFAULT_CORPUS) -> dict[str, str]:
    """Load ``{filename: markdown}`` from a directory of .md files."""
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {path}")
    return {
        file.name: file.read_text(encoding="utf-8")
        for file in sorted(path.glob("*.md"))
    }
