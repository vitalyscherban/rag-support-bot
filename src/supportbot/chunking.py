"""Structure-aware markdown chunking.

Fixed-size chunking is the original sin of cheap RAG: it splits mid-sentence,
orphans code blocks from the prose that explains them, and strips the heading
that says which product the section is even about. You then compensate by
retrieving *more* chunks -- paying tokens to undo your own damage.

Splitting on heading boundaries and prefixing each chunk with its heading path
("Billing > Refunds > Partial refunds") makes a single chunk self-describing,
which is what lets the retriever get away with top_k=3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import CHUNKS
from .tokens import count_tokens

HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")
FENCE_RE = re.compile(r"^```")


@dataclass
class Chunk:
    text: str
    source: str
    heading_path: tuple[str, ...] = ()
    chunk_id: str = ""
    embedding: list[float] = field(default_factory=list)

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.heading_path)

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.breadcrumb}" if self.heading_path else self.source

    def render(self) -> str:
        """What actually lands in the prompt: breadcrumb + body."""
        header = f"[{self.citation}]"
        return f"{header}\n{self.text.strip()}"

    @property
    def tokens(self) -> int:
        return count_tokens(self.render())


def _split_sections(markdown: str) -> list[tuple[tuple[str, ...], list[str]]]:
    """Walk the document, tracking the live heading stack. Fences are opaque."""
    sections: list[tuple[tuple[str, ...], list[str]]] = []
    stack: list[str] = []
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        if any(line.strip() for line in body):
            sections.append((tuple(stack), list(body)))
        body.clear()

    for line in markdown.splitlines():
        if FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            body.append(line)
            continue
        if in_fence:
            body.append(line)
            continue

        heading = HEADING_RE.match(line)
        if heading:
            flush()
            depth = len(heading.group("hashes"))
            del stack[depth - 1 :]
            while len(stack) < depth - 1:
                stack.append("")
            stack.append(heading.group("title"))
            continue

        body.append(line)

    flush()
    return sections


def _split_oversized(text: str, limit: int) -> list[str]:
    """Break a too-large section on blank lines, never mid-paragraph.

    Falls back to sentence boundaries, and finally to a hard word split, so a
    single pathological paragraph cannot smuggle an unbounded chunk into the
    index -- which would defeat the whole context ceiling downstream.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    parts: list[str] = []
    current: list[str] = []

    for paragraph in paragraphs:
        if count_tokens(paragraph) > limit:
            if current:
                parts.append("\n\n".join(current))
                current = []
            parts.extend(_split_paragraph(paragraph, limit))
            continue

        candidate = "\n\n".join([*current, paragraph])
        if current and count_tokens(candidate) > limit:
            parts.append("\n\n".join(current))
            current = [paragraph]
        else:
            current.append(paragraph)

    if current:
        parts.append("\n\n".join(current))
    return [part for part in parts if part.strip()]


def _split_paragraph(paragraph: str, limit: int) -> list[str]:
    """Sentence-wise split, with a hard word split for sentences that are still huge."""
    units = re.split(r"(?<=[.!?])\s+", paragraph)
    parts: list[str] = []
    current: list[str] = []

    for unit in units:
        if count_tokens(unit) > limit:
            if current:
                parts.append(" ".join(current))
                current = []
            parts.extend(_split_words(unit, limit))
            continue
        candidate = " ".join([*current, unit])
        if current and count_tokens(candidate) > limit:
            parts.append(" ".join(current))
            current = [unit]
        else:
            current.append(unit)

    if current:
        parts.append(" ".join(current))
    return parts


def _split_words(text: str, limit: int) -> list[str]:
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if count_tokens(" ".join(current)) >= limit:
            parts.append(" ".join(current))
            current = []
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_markdown(markdown: str, source: str) -> list[Chunk]:
    """Split one markdown document into self-describing chunks."""
    chunks: list[Chunk] = []

    for heading_path, lines in _split_sections(markdown):
        text = "\n".join(lines).strip()
        if not text:
            continue
        path = tuple(part for part in heading_path if part)
        for part in _split_oversized(text, CHUNKS.max_tokens):
            chunks.append(Chunk(text=part, source=source, heading_path=path))

    merged = _merge_runts(chunks)
    for index, chunk in enumerate(merged):
        chunk.chunk_id = f"{source}::{index}"
    return merged


def _merge_runts(chunks: list[Chunk]) -> list[Chunk]:
    """Fold a tiny section forward into the section it introduces.

    A lone "## Overview" with one sentence under it is retrieval noise on its
    own, but useful glued to what follows. Two rules keep this honest:

    * **Merge forward, not backward.** The following chunk's heading path is
      kept because it is the more specific of the two. Merging backwards would
      relabel a nested section with its parent's breadcrumb.
    * **At most one step.** Without this, a document whose sections are all
      short cascades into a single chunk carrying the deepest breadcrumb -- a
      label that is simply false for most of its content, and a chunk too
      coarse to retrieve precisely.
    """
    out: list[Chunk] = []
    pending: Chunk | None = None

    for chunk in chunks:
        if pending is not None:
            combined = f"{pending.text}\n\n{chunk.text}"
            mergeable = (
                pending.heading_path == chunk.heading_path[: len(pending.heading_path)]
                and count_tokens(combined) <= CHUNKS.max_tokens
            )
            if mergeable:
                out.append(
                    Chunk(
                        text=combined,
                        source=chunk.source,
                        heading_path=chunk.heading_path,
                    )
                )
                pending = None
                continue
            out.append(pending)
            pending = None

        if count_tokens(chunk.text) < CHUNKS.min_tokens:
            pending = chunk
            continue
        out.append(chunk)

    if pending is not None:
        # A trailing runt has nothing to merge forward into; attach it to the
        # previous chunk when that is cheap, otherwise let it stand alone.
        if out and count_tokens(out[-1].text + pending.text) <= CHUNKS.max_tokens:
            out[-1].text = f"{out[-1].text}\n\n{pending.text}"
        else:
            out.append(pending)

    return out


def chunk_corpus(documents: dict[str, str]) -> list[Chunk]:
    """Chunk a ``{source: markdown}`` mapping into one flat list."""
    chunks: list[Chunk] = []
    for source, markdown in sorted(documents.items()):
        chunks.extend(chunk_markdown(markdown, source))
    return chunks
