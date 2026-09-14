"""Cache tests.

The interesting cases are the *rejections*. A cache that never hits is merely
slow; a cache that hits wrongly answers the customer's question with someone
else's answer, and does it confidently.
"""

from supportbot.cache import SemanticCache
from supportbot.config import CACHE


def test_paraphrase_hits(embeddings):
    cache = SemanticCache(embeddings)
    cache.store("how do I reset my password", "Use Forgot password.", ["auth.md"])
    assert cache.lookup("password reset steps?") is not None


def test_same_topic_different_question_is_rejected(embeddings):
    """Regression: the one failure mode that actually hurts.

    'What is the default rate limit' and 'How do I raise my rate limit' are the
    same topic with different answers. Serving one for the other is worse than
    a cache miss, so the lookup must refuse.
    """
    cache = SemanticCache(embeddings)
    cache.store("What is the default API rate limit?", "1,000 rpm per key.", ["rate-limits.md"])
    assert cache.lookup("How do I raise my rate limit?") is None


def test_lexical_gate_rejects_high_embedding_similarity(embeddings):
    """Exercise the second signal directly.

    With the embedding threshold dropped to zero, only the lexical gate can
    reject -- proving it is load-bearing rather than decorative.
    """
    cache = SemanticCache(embeddings, threshold=0.0)
    cache.store("What is the default API rate limit?", "1,000 rpm per key.", ["rate-limits.md"])
    assert cache.lookup("How do I raise my rate limit?") is None
    assert cache.rejected, "a lexical rejection should be recorded for tuning"
    _question, _matched, score, overlap = cache.rejected[0]
    assert overlap < CACHE.min_lexical_overlap <= 1.0
    assert score > 0.0


def test_unrelated_question_misses(embeddings):
    cache = SemanticCache(embeddings)
    cache.store("how do I get a refund", "Within 30 days.", ["billing.md"])
    assert cache.lookup("how do I reset my password") is None


def test_threshold_is_backend_aware(embeddings):
    """A threshold tuned for a trained model must not be reused for the fallback."""
    approximate = SemanticCache(embeddings)
    assert approximate.threshold == CACHE.approximate_hit_threshold

    class TrainedLike(type(embeddings)):
        is_approximate = False

    assert SemanticCache(TrainedLike()).threshold == CACHE.semantic_hit_threshold


def test_stats_track_hit_rate(embeddings):
    cache = SemanticCache(embeddings)
    cache.store("how do I reset my password", "Use Forgot password.", ["auth.md"])
    cache.lookup("password reset steps?")
    cache.lookup("something entirely different about invoices")
    assert cache.stats.lookups == 2
    assert cache.stats.hits == 1
    assert cache.stats.hit_rate == 0.5


def test_eviction_respects_max_entries(embeddings):
    cache = SemanticCache(embeddings, max_entries=3)
    for index in range(6):
        cache.store(f"question about topic {index}", f"answer {index}", [])
    assert len(cache.entries) <= 3
