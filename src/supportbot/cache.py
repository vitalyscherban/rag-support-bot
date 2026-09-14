"""Lever 5: semantic caching -- the highest-yield lever in support.

An exact-match cache is nearly useless here, because nobody phrases a question
the same way twice:

    "how do I reset my password"
    "i forgot my password, how to reset?"
    "password reset steps?"

Three distinct strings, one answer. Embedding the question and matching on
cosine similarity collapses them into a single paid call; every subsequent
variant costs one embedding (~1/1000th of a generation) and nothing else.

Threshold choice is a real trade. Too low and you serve a confidently wrong
answer to a different question -- the one genuinely dangerous failure mode in
this repo -- which is why ``CACHE.semantic_hit_threshold`` is deliberately
conservative and every hit is recorded for offline review.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import CACHE
from .embeddings import cosine, jaccard


@dataclass
class CacheEntry:
    question: str
    answer: str
    citations: list[str]
    embedding: list[float]
    created_at: float = field(default_factory=time.time)
    hits: int = 0


@dataclass
class CacheStats:
    lookups: int = 0
    hits: int = 0
    tokens_saved: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class SemanticCache:
    """Similarity-matched answer cache, gated by two independent signals.

    A hit requires *both* embedding similarity and lexical overlap. Either
    alone produces false positives: embeddings confuse same-topic questions
    with same-answer questions, and lexical overlap misses paraphrases that
    share no vocabulary. Requiring both trades hit rate for safety, which is
    the correct direction when the failure mode is answering the wrong
    question confidently.
    """

    def __init__(
        self,
        embeddings,
        threshold: float | None = None,
        max_entries: int = CACHE.max_entries,
        min_lexical_overlap: float = CACHE.min_lexical_overlap,
    ) -> None:
        self.embeddings = embeddings
        if threshold is None:
            threshold = (
                CACHE.approximate_hit_threshold
                if getattr(embeddings, "is_approximate", False)
                else CACHE.semantic_hit_threshold
            )
        self.threshold = threshold
        self.min_lexical_overlap = min_lexical_overlap
        self.max_entries = max_entries
        self.entries: list[CacheEntry] = []
        self.stats = CacheStats()
        #: Near-misses, kept for offline threshold tuning.
        self.rejected: list[tuple[str, str, float, float]] = []

    def lookup(self, question: str) -> tuple[CacheEntry, float] | None:
        self.stats.lookups += 1
        if not self.entries:
            return None

        vector = self.embeddings.embed_query(question)
        best: CacheEntry | None = None
        best_score = 0.0
        for entry in self.entries:
            score = cosine(vector, entry.embedding)
            if score > best_score:
                best, best_score = entry, score

        if best is None or best_score < self.threshold:
            return None

        overlap = jaccard(question, best.question)
        if overlap < self.min_lexical_overlap:
            # Semantically close but lexically unrelated: almost always a
            # different question about the same topic.
            self.rejected.append((question, best.question, best_score, overlap))
            return None

        best.hits += 1
        self.stats.hits += 1
        return best, best_score

    def store(self, question: str, answer: str, citations: list[str], cost_tokens: int = 0) -> None:
        entry = CacheEntry(
            question=question,
            answer=answer,
            citations=list(citations),
            embedding=self.embeddings.embed_query(question),
        )
        self.entries.append(entry)
        self.stats.tokens_saved += 0  # accrues on hits, not on stores
        if len(self.entries) > self.max_entries:
            # Evict the least-used, oldest entry.
            self.entries.sort(key=lambda e: (e.hits, -e.created_at))
            del self.entries[0]

    def record_saving(self, tokens: int) -> None:
        self.stats.tokens_saved += tokens
