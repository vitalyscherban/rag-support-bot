"""Lever ablation: turn each optimization off and measure what it was worth.

    python benchmarks/ablation.py

A combined "88% saved" headline hides which levers earn their complexity. This
replays the same traffic with one lever disabled at a time; the delta is that
lever's contribution *in the presence of the others*, which is the only number
worth acting on. Levers overlap, so the individual savings do not sum to the
total -- that is expected, not an error.

Two metrics are reported, because they do not move together:

* **prompt tokens** -- what you are billed for at generation time.
* **retrieval calls** -- embedding + vector search round-trips. Routing barely
  moves tokens (the relevance floor already discards greetings) but eliminates
  these entirely, which is a latency and embedding-cost win rather than a
  generation-cost one. Reporting only tokens would make routing look worthless.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from supportbot.compress import assemble_context, compress_chunk, keyword_guard  # noqa: E402
from supportbot.config import RETRIEVAL  # noqa: E402
from supportbot.corpus import load_corpus  # noqa: E402
from supportbot.pipeline import SupportBot  # noqa: E402
from supportbot.prompts import SYSTEM_PROMPT  # noqa: E402
from supportbot.rerank import (  # noqa: E402
    drop_low_relevance,
    drop_near_duplicates,
    maximal_marginal_relevance,
)
from supportbot.router import Route, classify  # noqa: E402
from supportbot.tokens import count_tokens, savings  # noqa: E402
from traffic import TRAFFIC  # noqa: E402


@dataclass
class Result:
    tokens: int = 0
    retrievals: int = 0


def run(
    documents,
    *,
    routing: bool = True,
    cache: bool = True,
    floor: bool = True,
    dedupe: bool = True,
    compress: bool = True,
    ceiling: bool = True,
    top_k: int = RETRIEVAL.top_k,
) -> Result:
    """Replay the traffic with a subset of levers enabled."""
    bot = SupportBot(documents)
    result = Result()

    for question in TRAFFIC:
        if routing and classify(question).route is Route.TRIVIAL:
            continue

        if cache and bot.cache.lookup(question) is not None:
            continue

        result.retrievals += 1
        candidates = bot.store.search(question, k=RETRIEVAL.fetch_k)

        selected = candidates
        if floor:
            selected = drop_low_relevance(selected)
        if dedupe:
            selected = drop_near_duplicates(selected, threshold=bot.dedupe_threshold)
            selected = maximal_marginal_relevance(selected, k=top_k)
        else:
            selected = selected[:top_k]
        if floor:
            selected = [(c, s) for c, s in selected if keyword_guard(c, question)]

        if not selected:
            continue

        chunks = [c for c, _ in selected]
        if compress:
            query_vector = bot.embeddings.embed_query(question)
            chunks = [compress_chunk(c, query_vector, bot.embeddings) for c in chunks]

        if ceiling:
            context = assemble_context(chunks)
        else:
            context = "\n\n---\n\n".join(c.render() for c in chunks)

        result.tokens += (
            count_tokens(SYSTEM_PROMPT) + count_tokens(context) + count_tokens(question)
        )

        if cache:
            bot.cache.store(question, "answer", [c.citation for c in chunks])

    return result


def main() -> None:
    documents = load_corpus()

    naive = run(
        documents,
        routing=False,
        cache=False,
        floor=False,
        dedupe=False,
        compress=False,
        ceiling=False,
        top_k=RETRIEVAL.fetch_k,
    )
    full = run(documents)

    ablations = {
        "no routing": run(documents, routing=False),
        "no semantic cache": run(documents, cache=False),
        "no relevance floor": run(documents, floor=False),
        "no dedupe/MMR": run(documents, dedupe=False),
        "no sentence compression": run(documents, compress=False),
        "no context ceiling": run(documents, ceiling=False),
        "top_k=10 instead of 3": run(documents, top_k=10),
    }

    print(f"traffic: {len(TRAFFIC)} queries\n")
    header = (
        f"{'configuration':<25} {'tokens':>8} {'vs naive':>9} "
        f"{'retrievals':>11} {'cost of removing':>18}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'naive (top-20 raw)':<25} {naive.tokens:>8,} {'--':>9} "
        f"{naive.retrievals:>11} {'--':>18}"
    )
    print(
        f"{'all levers on':<25} {full.tokens:>8,} "
        f"{savings(naive.tokens, full.tokens):>9} {full.retrievals:>11} {'--':>18}"
    )
    print("-" * len(header))

    for name, result in sorted(ablations.items(), key=lambda kv: -kv[1].tokens):
        delta_tokens = result.tokens - full.tokens
        delta_retrievals = result.retrievals - full.retrievals
        if delta_tokens:
            cost = f"+{delta_tokens / full.tokens * 100:.0f}% tokens"
        elif delta_retrievals:
            cost = f"+{delta_retrievals} retrievals"
        else:
            cost = "negligible"
        print(
            f"{name:<25} {result.tokens:>8,} "
            f"{savings(naive.tokens, result.tokens):>9} {result.retrievals:>11} {cost:>18}"
        )

    print(
        "\nRead the last column as: removing this lever costs you that much extra.\n"
        "Levers overlap, so individual numbers do not sum to the total.\n"
        "Routing shows up in retrievals, not tokens -- the relevance floor already\n"
        "discards greetings, so routing's win is latency and embedding spend."
    )


if __name__ == "__main__":
    main()
