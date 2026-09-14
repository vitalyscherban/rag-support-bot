"""Token counting, shared by the pipeline, the tests and the benchmark."""

from __future__ import annotations

from typing import Iterable

try:  # pragma: no cover - environment dependent
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENCODING = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, len(text) // 4)


def count_all(texts: Iterable[str]) -> int:
    return sum(count_tokens(t) for t in texts)


def savings(before: int, after: int) -> str:
    if before <= 0:
        return "n/a"
    return f"{(1 - after / before) * 100:.1f}%"
