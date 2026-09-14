"""Lever 1: decide whether to retrieve at all, and which model to use.

Real support traffic is not uniform. A meaningful slice is greetings, thanks,
and chit-chat that needs no documents whatsoever -- yet a naive bot embeds the
query, retrieves 20 chunks and calls the flagship model to say "you're welcome".

Routing happens before retrieval, so a correctly-routed trivial query costs
roughly a hundred tokens instead of several thousand. Classification is done
with rules, not a model: paying an LLM to decide whether to pay an LLM is a
tax that eats the saving it was meant to produce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .embeddings import stem, tokenize


class Route(str, Enum):
    #: Greetings, thanks, acknowledgements. No retrieval, canned reply.
    TRIVIAL = "trivial"
    #: A factual lookup. Retrieval, then the cheap model.
    SIMPLE = "simple"
    #: Multi-step, comparative or debugging. Retrieval, then the smart model.
    COMPLEX = "complex"


# Stemmed at import time because ``tokenize`` stems its output: a literal
# "cheers" here would never match the stemmed token "cheer", silently routing
# greetings into full retrieval. Deriving the set removes that whole class of
# drift between the tokenizer and its consumers.
GREETINGS = frozenset(
    stem(word)
    for word in (
        "hi", "hello", "hey", "thanks", "thank", "ty", "bye", "goodbye",
        "morning", "afternoon", "cheers", "ok", "okay", "cool", "great",
    )
)

COMPLEX_MARKERS = (
    "why", "compare", "difference", "versus", " vs ", "trade-off", "tradeoff",
    "debug", "troubleshoot", "root cause", "architecture", "design",
    "best practice", "migrate", "should i", "which is better", "explain how",
)

STEP_MARKERS = re.compile(r"\b(first|then|after that|and also|as well as)\b")


@dataclass(frozen=True)
class Decision:
    route: Route
    reason: str

    @property
    def needs_retrieval(self) -> bool:
        return self.route is not Route.TRIVIAL

    @property
    def model_tier(self) -> str:
        return "smart" if self.route is Route.COMPLEX else "cheap"


def classify(query: str) -> Decision:
    """Route a query. Deterministic, ~zero cost, easy to unit test."""
    text = query.lower().strip()
    words = tokenize(text)

    if not words:
        return Decision(Route.TRIVIAL, "empty query")

    # Short and made only of social tokens -> nothing to look up.
    if len(words) <= 4 and all(word in GREETINGS for word in words):
        return Decision(Route.TRIVIAL, "greeting or acknowledgement")

    padded = f" {text} "
    if any(marker in padded for marker in COMPLEX_MARKERS):
        return Decision(Route.COMPLEX, "comparative or causal question")

    if text.count("?") > 1 or STEP_MARKERS.search(text):
        return Decision(Route.COMPLEX, "multi-part question")

    if len(words) > 25:
        return Decision(Route.COMPLEX, "long question, likely multi-faceted")

    return Decision(Route.SIMPLE, "single factual lookup")


CANNED_REPLIES = {
    stem(key): value
    for key, value in {
        "hi": "Hi! Ask me anything about the docs.",
        "hello": "Hello! What can I help you find in the documentation?",
        "hey": "Hey! What would you like to know?",
        "thanks": "Happy to help!",
        "thank": "Happy to help!",
        "bye": "Bye! Come back any time.",
        "goodbye": "Bye! Come back any time.",
    }.items()
}


def canned_reply(query: str) -> str:
    for word in tokenize(query):
        if word in CANNED_REPLIES:
            return CANNED_REPLIES[word]
    return "Ask me anything about the docs."
