"""Retrieval tests: the savings must not cost us the answer.

Every test here is a fidelity test disguised as an efficiency test. Cutting
tokens is trivial if you are allowed to cut the right chunk too.
"""

from supportbot.chunking import chunk_corpus
from supportbot.compress import assemble_context, compress_chunk, split_sentences
from supportbot.config import RETRIEVAL
from supportbot.rerank import drop_low_relevance, drop_near_duplicates, rerank
from supportbot.store import VectorStore
from supportbot.tokens import count_tokens


def test_dedupe_collapses_faq_restatements(bot):
    """faq.md deliberately restates billing.md; both must not reach the prompt.

    'Refunds post to the original payment method within 5-10 business days'
    appears in both files and scores ~0.74 with the offline embedding.
    """
    candidates = bot.store.search("partial refund annual plan", k=RETRIEVAL.fetch_k)
    deduped = drop_near_duplicates(candidates, threshold=bot.dedupe_threshold)
    assert len(deduped) < len(candidates)


def test_dedupe_threshold_is_backend_aware(bot, embeddings):
    """A trained-model threshold would make dedupe a silent no-op offline."""
    assert bot.dedupe_threshold == RETRIEVAL.approximate_dedupe_threshold
    candidates = bot.store.search("partial refund annual plan", k=RETRIEVAL.fetch_k)
    assert len(drop_near_duplicates(candidates, threshold=0.92)) == len(candidates)


def test_dedupe_keeps_the_higher_scoring_copy(bot):
    candidates = bot.store.search("how long does a refund take", k=RETRIEVAL.fetch_k)
    deduped = drop_near_duplicates(candidates, threshold=bot.dedupe_threshold)
    kept_ids = {chunk.chunk_id for chunk, _ in deduped}
    scores = {chunk.chunk_id: score for chunk, score in candidates}
    for chunk, score in candidates:
        if chunk.chunk_id not in kept_ids:
            assert max(scores[i] for i in kept_ids) >= score


def test_relevance_floor_drops_noise(bot):
    candidates = bot.store.search("partial refund annual plan", k=RETRIEVAL.fetch_k)
    filtered = drop_low_relevance(candidates, floor=0.15)
    assert len(filtered) < len(candidates)
    assert all(score >= 0.15 for _, score in filtered)


def test_rerank_returns_at_most_top_k(bot):
    selected = rerank(bot.store.search("how do refunds work", k=RETRIEVAL.fetch_k))
    assert len(selected) <= RETRIEVAL.top_k


def test_rerank_keeps_the_right_document(bot):
    selected = rerank(bot.store.search("partial refund on an annual plan", k=RETRIEVAL.fetch_k))
    sources = {chunk.source for chunk, _ in selected}
    assert "billing.md" in sources or "faq.md" in sources


def test_retrieval_finds_rate_limit_answer(bot):
    selected = rerank(bot.store.search("default requests per minute limit", k=RETRIEVAL.fetch_k))
    text = " ".join(chunk.text for chunk, _ in selected)
    assert "1,000 requests per minute" in text


def test_retrieval_finds_password_reset(bot):
    selected = rerank(bot.store.search("reset password forgot link expiry", k=RETRIEVAL.fetch_k))
    text = " ".join(chunk.text for chunk, _ in selected).lower()
    assert "forgot password" in text or "reset link" in text


def test_compression_preserves_document_order(bot, embeddings):
    """Reordering by score shreds numbered procedures; order must survive."""
    chunk = max(chunk_corpus({"d.md": _ORDERED_DOC}), key=lambda c: len(c.text))
    query_vector = embeddings.embed_query("rotate key grace period immediate revoke")
    compressed = compress_chunk(chunk, query_vector, embeddings, max_sentences=2)
    original = split_sentences(chunk.text)
    kept = [s for s in split_sentences(compressed.text) if s != "..."]
    positions = [original.index(s) for s in kept if s in original]
    assert positions == sorted(positions)


def test_compression_reduces_tokens(bot, embeddings):
    store = VectorStore(embeddings).add(chunk_corpus({"d.md": _ORDERED_DOC}))
    chunk = max(store.chunks, key=lambda c: count_tokens(c.text))
    query_vector = embeddings.embed_query("grace period")
    compressed = compress_chunk(chunk, query_vector, embeddings, max_sentences=1)
    assert count_tokens(compressed.text) < count_tokens(chunk.text)


def test_assemble_context_respects_ceiling(bot):
    selected = rerank(bot.store.search("refund", k=RETRIEVAL.fetch_k))
    context = assemble_context([c for c, _ in selected], max_tokens=80)
    assert count_tokens(context) <= 80


_ORDERED_DOC = """# Keys

## Rotation

Rotate with POST /v1/keys/{key_id}/rotate. The old key remains valid for a
24-hour grace period so deployments can roll forward. Pass immediate true to
revoke the old key instantly during an incident. Audit entries are written for
both the creation and the revocation. Contact support if the rotation stalls.
"""
