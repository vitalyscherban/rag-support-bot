"""Lever 2: cut 20 candidates to 3 before the prompt ever sees them.

Naive RAG sends the top-20 chunks. In a real docs corpus those 20 contain:

* near-duplicates (the same paragraph in the tutorial, the API reference and
  the changelog);
* topical-but-useless neighbours that rank well because they share vocabulary;
* a long tail scoring barely above noise.

Three ordered filters remove each class. All three are pure vector arithmetic --
no model call, so the reduction itself costs zero tokens.
"""

from __future__ import annotations

from .chunking import Chunk
from .config import RETRIEVAL
from .embeddings import cosine


def drop_near_duplicates(
    scored: list[tuple[Chunk, float]],
    threshold: float = RETRIEVAL.dedupe_threshold,
) -> list[tuple[Chunk, float]]:
    """Greedy dedupe, highest-scoring copy wins.

    Order matters: because the list arrives sorted by relevance, the survivor of
    any duplicate pair is always the best-ranked phrasing.
    """
    kept: list[tuple[Chunk, float]] = []
    for chunk, score in scored:
        if any(cosine(chunk.embedding, other.embedding) >= threshold for other, _ in kept):
            continue
        kept.append((chunk, score))
    return kept


def drop_low_relevance(
    scored: list[tuple[Chunk, float]],
    floor: float = RETRIEVAL.min_relevance,
) -> list[tuple[Chunk, float]]:
    """Never pad to top_k with junk.

    A fixed k forces the retriever to return k things whether or not k things
    are relevant. Enforcing a floor is what lets the bot say "not in the docs"
    instead of confabulating from a weak match.
    """
    return [(chunk, score) for chunk, score in scored if score >= floor]


def maximal_marginal_relevance(
    scored: list[tuple[Chunk, float]],
    k: int = RETRIEVAL.top_k,
    diversity: float = 0.3,
) -> list[tuple[Chunk, float]]:
    """Trade a little relevance for coverage.

    Picking the top-3 by score alone often returns three angles on the same
    sentence. MMR penalises a candidate by its similarity to what is already
    selected, so the three chunks tend to cover three different facets --
    higher answer quality at identical token cost.
    """
    if not scored:
        return []
    selected: list[tuple[Chunk, float]] = [scored[0]]
    pool = list(scored[1:])

    while pool and len(selected) < k:
        best_index, best_value = 0, float("-inf")
        for index, (chunk, score) in enumerate(pool):
            redundancy = max(
                cosine(chunk.embedding, chosen.embedding) for chosen, _ in selected
            )
            value = (1 - diversity) * score - diversity * redundancy
            if value > best_value:
                best_index, best_value = index, value
        selected.append(pool.pop(best_index))

    return selected


def rerank(
    scored: list[tuple[Chunk, float]],
    k: int = RETRIEVAL.top_k,
    dedupe_threshold: float | None = None,
) -> list[tuple[Chunk, float]]:
    """The full funnel: relevance floor -> dedupe -> MMR selection."""
    filtered = drop_low_relevance(scored)
    threshold = (
        RETRIEVAL.dedupe_threshold if dedupe_threshold is None else dedupe_threshold
    )
    deduped = drop_near_duplicates(filtered, threshold=threshold)
    return maximal_marginal_relevance(deduped, k=k)
