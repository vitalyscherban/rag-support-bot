"""Naive vs optimized token accounting. Runs fully offline -- no API key.

    python benchmarks/compare.py

Replays a realistic support traffic mix through both pipelines and reports
tokens, cost, and which lever produced the saving.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from supportbot.config import PRICE_PER_1K, RETRIEVAL  # noqa: E402
from supportbot.corpus import load_corpus  # noqa: E402
from supportbot.pipeline import SupportBot  # noqa: E402
from supportbot.router import Route, classify  # noqa: E402
from supportbot.tokens import savings  # noqa: E402
from traffic import TRAFFIC  # noqa: E402

QUERIES_PER_DAY = 2_000


def main() -> None:
    documents = load_corpus()
    bot = SupportBot(documents)

    print(f"corpus: {len(documents)} documents -> {len(bot.store)} chunks")
    print(f"traffic: {len(TRAFFIC)} queries\n")

    naive_total = 0
    optimized_total = 0
    by_route: dict[str, list[int]] = {}
    cache_hits = 0

    for question in TRAFFIC:
        naive = bot.ask_naive(question)
        naive_total += naive.prompt_tokens

        answer = bot.ask(question)
        optimized_total += answer.prompt_tokens
        cache_hits += int(answer.cache_hit)

        key = "cache hit" if answer.cache_hit else answer.route
        by_route.setdefault(key, []).append(answer.prompt_tokens)

    print("WHERE THE TOKENS WENT (optimized pipeline)")
    print(f"{'exit stage':<14} {'queries':>8} {'avg tokens':>12} {'total':>10}")
    print("-" * 48)
    for key in ("trivial", "cache hit", "simple", "complex"):
        tokens = by_route.get(key)
        if not tokens:
            continue
        print(
            f"{key:<14} {len(tokens):>8} "
            f"{sum(tokens) / len(tokens):>12,.0f} {sum(tokens):>10,}"
        )

    print("\nNAIVE vs OPTIMIZED")
    print(f"{'':<22} {'naive':>10} {'optimized':>10} {'saved':>9}")
    print("-" * 54)
    print(
        f"{'prompt tokens':<22} {naive_total:>10,} {optimized_total:>10,} "
        f"{savings(naive_total, optimized_total):>9}"
    )

    per_query_naive = naive_total / len(TRAFFIC)
    per_query_opt = optimized_total / len(TRAFFIC)
    print(
        f"{'per query':<22} {per_query_naive:>10,.0f} {per_query_opt:>10,.0f} "
        f"{savings(naive_total, optimized_total):>9}"
    )

    # Naive sends everything to the flagship model; optimized routes by need.
    naive_cost = naive_total / 1000 * PRICE_PER_1K["smart"]
    optimized_cost = optimized_total / 1000 * PRICE_PER_1K["cheap"]
    print(
        f"{'cost (this run)':<22} {'$' + format(naive_cost, '.4f'):>10} "
        f"{'$' + format(optimized_cost, '.4f'):>10} "
        f"{savings(int(naive_cost * 1e6), int(optimized_cost * 1e6)):>9}"
    )

    scale = QUERIES_PER_DAY / len(TRAFFIC) * 30
    print(
        f"\nat {QUERIES_PER_DAY:,} queries/day: "
        f"${naive_cost * scale:,.2f}/mo -> ${optimized_cost * scale:,.2f}/mo"
    )

    print(
        f"\ncache hit rate {bot.cache.stats.hit_rate:.0%} "
        f"({cache_hits}/{bot.cache.stats.lookups} lookups)"
    )
    routes = {r.value: 0 for r in Route}
    for question in TRAFFIC:
        routes[classify(question).route.value] += 1
    print(f"routing: {routes}")
    print(f"retrieval: fetch_k={RETRIEVAL.fetch_k} -> top_k={RETRIEVAL.top_k}")


if __name__ == "__main__":
    main()
