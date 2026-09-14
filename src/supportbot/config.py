"""Every token knob in one place.

Each value is a recall-vs-cost trade. The benchmark sweeps them; the test suite
pins the fidelity floor below which cutting further starts losing answers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalBudget:
    # Candidates pulled from the vector store before any filtering.
    fetch_k: int = 20
    # Chunks surviving dedupe + rerank, i.e. what the prompt actually pays for.
    top_k: int = 3
    # Cosine similarity above which two chunks are treated as duplicates.
    # Like the cache thresholds, this is calibrated per embedding backend: the
    # offline hashing embedding rates a genuine restatement at ~0.74, so the
    # trained-model value of 0.92 would never fire and dedupe would silently
    # become a no-op.
    dedupe_threshold: float = 0.92
    approximate_dedupe_threshold: float = 0.70
    # Chunks scoring below this are dropped even if top_k is unfilled.
    min_relevance: float = 0.15
    # Sentences kept per chunk during extractive compression.
    max_sentences_per_chunk: int = 4
    # Hard ceiling on the assembled context block.
    max_context_tokens: int = 1_200

    def dedupe_for(self, embeddings) -> float:
        return (
            self.approximate_dedupe_threshold
            if getattr(embeddings, "is_approximate", False)
            else self.dedupe_threshold
        )


@dataclass(frozen=True)
class CacheBudget:
    # Cosine similarity at which an incoming question is considered a repeat of
    # one already answered. Support traffic is extremely repetitive, so this is
    # the single highest-yield lever in the whole pipeline.
    #
    # Tuned for a trained embedding (text-embedding-3-small). The offline
    # hashing fallback has a completely different similarity distribution and
    # gets `approximate_hit_threshold` instead -- reusing this number there
    # would serve wrong answers with total confidence.
    semantic_hit_threshold: float = 0.93
    approximate_hit_threshold: float = 0.72
    # Second signal: a hit must also share this fraction of content words.
    # Embeddings alone rate "what is the default rate limit" and "how do I
    # raise my rate limit" as near-identical; they have different answers.
    min_lexical_overlap: float = 0.5
    max_entries: int = 1_000


@dataclass(frozen=True)
class ChunkBudget:
    # Chunks are built from markdown structure, then clamped to these bounds.
    target_tokens: int = 220
    max_tokens: int = 400
    # Tiny fragments (stub sections, lone headings) are merged forward.
    min_tokens: int = 40


@dataclass(frozen=True)
class Models:
    cheap: str = os.getenv("CHEAP_MODEL", "gpt-4o-mini")
    smart: str = os.getenv("SMART_MODEL", "gpt-4o")
    embedding: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


RETRIEVAL = RetrievalBudget()
CACHE = CacheBudget()
CHUNKS = ChunkBudget()
MODELS = Models()

# Pricing used only by the benchmark to turn token counts into dollars.
PRICE_PER_1K = {"cheap": 0.00015, "smart": 0.0025}
