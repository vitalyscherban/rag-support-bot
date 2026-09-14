"""Embeddings with a deterministic offline fallback.

``HashingEmbeddings`` is a real (if crude) bag-of-words embedding: it hashes
tokens into a fixed-width vector with sublinear term weighting and L2
normalisation. It is not competitive with a trained model, but it is
deterministic, dependency-free and fast -- which means the whole retrieval
pipeline, its tests and its benchmark run offline in CI with no API key.

Swap in ``OpenAIEmbeddings`` for production; the interface is identical.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9_]+")

STOPWORDS = frozenset(
    """a an and are as at be but by for from has have how i if in is it its of on or
    that the this to was what when where which who why will with you your do does
    can could should would my me we our""".split()
)

# Deliberately crude suffix stripping. A real stemmer (Snowball) is better, but
# this collapses the pairs that actually matter in support text -- refund/refunds,
# key/keys, rotate/rotating -- with no dependency and no surprises.
_SUFFIXES = ("ing", "ies", "es", "ed", "s")


def stem(token: str) -> str:
    if len(token) <= 3:
        return token
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            base = token[: -len(suffix)]
            return base + "y" if suffix == "ies" else base
    return token


def tokenize(text: str) -> list[str]:
    return [
        stem(t)
        for t in TOKEN_RE.findall(text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


def jaccard(a: str, b: str) -> float:
    """Lexical overlap of content words -- a second opinion on similarity.

    Used to gate semantic cache hits. Embeddings alone confuse "what is the
    default rate limit" with "how do I raise my rate limit": same topic,
    different answers. Requiring lexical agreement too kills that failure mode.
    """
    set_a, set_b = set(tokenize(a)), set(tokenize(b))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class HashingEmbeddings:
    """Deterministic bag-of-words embedding compatible with LangChain's API.

    ``is_approximate`` tells calibration-sensitive consumers (the semantic
    cache) that this backend's similarity distribution is not that of a trained
    model, so thresholds tuned for one must not be reused for the other.
    """

    is_approximate = True

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def embed_query(self, text: str) -> list[float]:
        counts = Counter(tokenize(text))
        vector = [0.0] * self.dim
        for token, count in counts.items():
            # Sublinear TF damps repeated words so one shouty term cannot
            # dominate a chunk's direction.
            vector[self._bucket(token)] += 1.0 + math.log(count)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Inputs are pre-normalised, so this is a dot product."""
    return sum(x * y for x, y in zip(a, b))


def get_embeddings(offline: bool = True):
    """Return the offline embedding, or OpenAI's when explicitly asked for."""
    if offline:
        return HashingEmbeddings()
    from langchain_openai import OpenAIEmbeddings

    from .config import MODELS

    return OpenAIEmbeddings(model=MODELS.embedding)
