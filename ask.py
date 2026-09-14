"""CLI for the docs support bot.

    python ask.py "how do I request a partial refund?"
    python ask.py --interactive
    python ask.py --explain "what is the rate limit?"    # show the funnel

Runs offline with the hashing embedding and no generation unless OPENAI_API_KEY
is set, so you can inspect retrieval behaviour without spending anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from supportbot.config import RETRIEVAL  # noqa: E402
from supportbot.corpus import load_corpus  # noqa: E402
from supportbot.pipeline import SupportBot  # noqa: E402
from supportbot.rerank import rerank  # noqa: E402
from supportbot.router import classify  # noqa: E402
from supportbot.tokens import count_tokens  # noqa: E402


def build_bot(corpus: Path, online: bool) -> SupportBot:
    documents = load_corpus(corpus)
    llm = None
    embeddings = None

    if online:
        from langchain_openai import ChatOpenAI

        from supportbot.config import MODELS
        from supportbot.embeddings import get_embeddings

        llm = ChatOpenAI(model=MODELS.cheap, temperature=0)
        embeddings = get_embeddings(offline=False)

    return SupportBot(documents, llm=llm, embeddings=embeddings)


def explain(bot: SupportBot, question: str) -> None:
    """Show each stage of the funnel and what it removed."""
    decision = classify(question)
    print(f"  route          {decision.route.value}  ({decision.reason})")
    print(f"  model tier     {decision.model_tier}")

    if not decision.needs_retrieval:
        print("  retrieval      skipped")
        return

    candidates = bot.store.search(question, k=RETRIEVAL.fetch_k)
    raw_tokens = count_tokens("\n\n".join(c.render() for c, _ in candidates))
    selected = rerank(candidates, k=RETRIEVAL.top_k, dedupe_threshold=bot.dedupe_threshold)

    print(f"  fetched        {len(candidates)} chunks / {raw_tokens:,} tokens")
    print(f"  after rerank   {len(selected)} chunks")
    for chunk, score in selected:
        print(f"                 {score:.3f}  {chunk.citation}")


def ask_once(bot: SupportBot, question: str, show_funnel: bool) -> None:
    if show_funnel:
        print(f"\nQ: {question}")
        explain(bot, question)

    answer = bot.ask(question)

    print(f"\n{answer.text}")
    if answer.citations:
        print("\nsources:")
        for citation in answer.citations:
            print(f"  - {citation}")

    detail = f"route={answer.route} tokens={answer.prompt_tokens:,}"
    if answer.cache_hit:
        detail += " (cache hit)"
    elif answer.naive_prompt_tokens:
        detail += f" vs naive {answer.naive_prompt_tokens:,} ({answer.savings_pct:.0f}% saved)"
    print(f"\n[{detail}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="*", help="the question to ask")
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--explain", action="store_true", help="show the retrieval funnel")
    parser.add_argument(
        "--online", action="store_true", help="use OpenAI embeddings and generation"
    )
    args = parser.parse_args()

    if args.online and not os.getenv("OPENAI_API_KEY"):
        print("--online requires OPENAI_API_KEY", file=sys.stderr)
        return 2

    bot = build_bot(args.corpus, args.online)
    if not args.online:
        print("[offline mode: retrieval is real, generation is stubbed]")
    print(f"[{len(bot.store)} chunks indexed]")

    if args.interactive:
        print("Ask a question, or Ctrl-C to quit.\n")
        try:
            while True:
                question = input("> ").strip()
                if question:
                    ask_once(bot, question, args.explain)
                    print()
        except (KeyboardInterrupt, EOFError):
            print("\nbye")
            hits = bot.cache.stats
            print(f"[cache {hits.hits}/{hits.lookups} hits, {hits.hit_rate:.0%}]")
        return 0

    if not args.question:
        parser.error("pass a question, or use --interactive")

    ask_once(bot, " ".join(args.question), args.explain)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
