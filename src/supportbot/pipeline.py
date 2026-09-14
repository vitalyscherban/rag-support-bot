"""The pipeline, wiring all five levers in cost order.

Ordering is the whole design. Each stage is cheaper than the one after it, so
the expensive stages only ever see traffic that survived the cheap ones:

    route (free)  ->  cache (1 embedding)  ->  retrieve (1 embedding)
                  ->  rerank (vector math) ->  compress (vector math)
                  ->  generate (the only stage that costs real money)

A trivial query exits at stage 1. A repeat question exits at stage 2. Only a
genuinely novel, substantive question pays for generation -- and when it does,
it pays for ~3 compressed chunks rather than 20 raw ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cache import SemanticCache
from .chunking import Chunk, chunk_corpus
from .compress import assemble_context, compress_chunk, keyword_guard
from .config import MODELS, RETRIEVAL
from .embeddings import get_embeddings
from .prompts import ANSWER_TEMPLATE, NO_CONTEXT_REPLY, SYSTEM_PROMPT
from .rerank import rerank
from .router import Route, canned_reply, classify
from .store import VectorStore
from .tokens import count_tokens


@dataclass
class Answer:
    text: str
    citations: list[str] = field(default_factory=list)
    route: str = ""
    cache_hit: bool = False
    prompt_tokens: int = 0
    context_tokens: int = 0
    model: str = ""
    #: Tokens a naive top-20-raw-chunks implementation would have spent.
    naive_prompt_tokens: int = 0

    @property
    def savings_pct(self) -> float:
        if not self.naive_prompt_tokens:
            return 0.0
        return (1 - self.prompt_tokens / self.naive_prompt_tokens) * 100


class SupportBot:
    """A docs Q&A bot that treats prompt tokens as a budget, not an afterthought."""

    def __init__(self, documents: dict[str, str], llm=None, embeddings=None) -> None:
        self.embeddings = embeddings or get_embeddings(offline=True)
        self.llm = llm
        self.store = VectorStore(self.embeddings).add(chunk_corpus(documents))
        self.cache = SemanticCache(self.embeddings)
        self.dedupe_threshold = RETRIEVAL.dedupe_for(self.embeddings)

    # -- stage 6 -----------------------------------------------------------
    def _generate(self, question: str, context: str, tier: str) -> str:
        if self.llm is None:
            # Offline mode: return the context so the pipeline stays testable
            # end to end without a key. Real deployments always pass an llm.
            return f"[offline] would answer from {count_tokens(context)} context tokens"
        prompt = ANSWER_TEMPLATE.format(context=context, question=question)
        model = self.llm if not hasattr(self.llm, "for_tier") else self.llm.for_tier(tier)
        from langchain_core.messages import HumanMessage, SystemMessage

        response = model.invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        )
        return response.content

    # -- the funnel --------------------------------------------------------
    def ask(self, question: str) -> Answer:
        decision = classify(question)

        # Stage 1: trivial traffic never touches the index or a model.
        if decision.route is Route.TRIVIAL:
            return Answer(
                text=canned_reply(question),
                route=decision.route.value,
                prompt_tokens=0,
                model="none",
            )

        # Stage 2: semantic cache. A hit costs one embedding, nothing else.
        cached = self.cache.lookup(question)
        if cached is not None:
            entry, _score = cached
            return Answer(
                text=entry.answer,
                citations=list(entry.citations),
                route=decision.route.value,
                cache_hit=True,
                prompt_tokens=0,
                model="cache",
            )

        # Stage 3: over-fetch cheaply, so the filters have something to work on.
        candidates = self.store.search(question, k=RETRIEVAL.fetch_k)
        naive_context = "\n\n".join(chunk.render() for chunk, _ in candidates)
        naive_tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(naive_context) + count_tokens(question)

        # Stage 4: floor -> dedupe -> MMR. Pure vector math, zero tokens.
        selected = rerank(candidates, k=RETRIEVAL.top_k, dedupe_threshold=self.dedupe_threshold)
        selected = [(c, s) for c, s in selected if keyword_guard(c, question)]

        if not selected:
            return Answer(
                text=NO_CONTEXT_REPLY,
                route=decision.route.value,
                prompt_tokens=0,
                model="none",
                naive_prompt_tokens=naive_tokens,
            )

        # Stage 5: drop the sentences inside each chunk that do not earn a place.
        query_vector = self.embeddings.embed_query(question)
        compressed: list[Chunk] = [
            compress_chunk(chunk, query_vector, self.embeddings) for chunk, _ in selected
        ]
        context = assemble_context(compressed)

        # Stage 6: generate, on the tier the router picked.
        model_name = MODELS.smart if decision.model_tier == "smart" else MODELS.cheap
        text = self._generate(question, context, decision.model_tier)

        citations = [chunk.citation for chunk in compressed]
        prompt_tokens = (
            count_tokens(SYSTEM_PROMPT) + count_tokens(context) + count_tokens(question)
        )

        self.cache.store(question, text, citations)
        self.cache.record_saving(naive_tokens - prompt_tokens)

        return Answer(
            text=text,
            citations=citations,
            route=decision.route.value,
            prompt_tokens=prompt_tokens,
            context_tokens=count_tokens(context),
            model=model_name,
            naive_prompt_tokens=naive_tokens,
        )

    def ask_naive(self, question: str) -> Answer:
        """The baseline: no routing, no cache, top-20 raw chunks. For the benchmark."""
        candidates = self.store.search(question, k=RETRIEVAL.fetch_k)
        context = "\n\n".join(chunk.render() for chunk, _ in candidates)
        tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(context) + count_tokens(question)
        return Answer(
            text=self._generate(question, context, "smart"),
            citations=[chunk.citation for chunk, _ in candidates],
            route="naive",
            prompt_tokens=tokens,
            context_tokens=count_tokens(context),
            model=MODELS.smart,
            naive_prompt_tokens=tokens,
        )
