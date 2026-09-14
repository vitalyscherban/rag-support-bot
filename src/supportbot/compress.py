"""Lever 3: extractive compression -- keep sentences, not chunks.

Even a well-targeted 220-token chunk is mostly scaffolding: a heading sentence,
a caveat about an unrelated plan tier, a "see also". Scoring each sentence
against the query and keeping the best few typically halves the context again
with no loss of answer quality.

This is the embedding-only version, which costs nothing. ``LLMChainExtractor``
does it better but spends a model call per chunk -- worth it only when chunks
are long and traffic is low.
"""

from __future__ import annotations

import re

from .chunking import Chunk
from .config import RETRIEVAL
from .embeddings import cosine, tokenize
from .tokens import count_tokens

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n(?=[-*\d])|\n{2,}")


def split_sentences(text: str) -> list[str]:
    """Sentence split that keeps list items and code fences intact."""
    blocks = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    sentences: list[str] = []
    for block in blocks:
        if block.startswith("```"):
            sentences.append(block.strip())
            continue
        sentences.extend(s.strip() for s in SENTENCE_RE.split(block) if s.strip())
    return sentences


def compress_chunk(
    chunk: Chunk,
    query_vector: list[float],
    embeddings,
    max_sentences: int = RETRIEVAL.max_sentences_per_chunk,
) -> Chunk:
    """Keep the ``max_sentences`` best sentences, in their original order.

    Preserving document order matters: reordering by score shreds the logic of
    a numbered procedure, and the model will happily invent steps to bridge it.
    """
    sentences = split_sentences(chunk.text)
    if len(sentences) <= max_sentences:
        return chunk

    vectors = embeddings.embed_documents(sentences)
    ranked = sorted(
        enumerate(zip(sentences, vectors)),
        key=lambda item: cosine(query_vector, item[1][1]),
        reverse=True,
    )
    keep_indexes = sorted(index for index, _ in ranked[:max_sentences])

    kept: list[str] = []
    previous: int | None = None
    for index in keep_indexes:
        if previous is not None and index > previous + 1:
            kept.append("...")
        kept.append(sentences[index])
        previous = index

    return Chunk(
        text=" ".join(kept),
        source=chunk.source,
        heading_path=chunk.heading_path,
        chunk_id=chunk.chunk_id,
        embedding=chunk.embedding,
    )


def keyword_guard(chunk: Chunk, query: str) -> bool:
    """Require at least one shared content word with the query.

    A pure-vector pipeline occasionally surfaces a chunk that is thematically
    adjacent but shares no actual terms -- classic embedding drift. This is a
    crude but effective backstop.
    """
    query_terms = set(tokenize(query))
    if not query_terms:
        return True
    return bool(query_terms & set(tokenize(chunk.text + " " + chunk.breadcrumb)))


def assemble_context(
    chunks: list[Chunk], max_tokens: int = RETRIEVAL.max_context_tokens
) -> str:
    """Join chunks under a hard token ceiling, dropping from the tail.

    The ceiling is the backstop that makes cost predictable: a pathological
    document can no longer blow the budget regardless of what ranked well.
    """
    parts: list[str] = []
    total = 0
    for chunk in chunks:
        rendered = chunk.render()
        cost = count_tokens(rendered)
        if total + cost > max_tokens:
            break
        parts.append(rendered)
        total += cost
    return "\n\n---\n\n".join(parts)
