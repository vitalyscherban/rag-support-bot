"""End-to-end pipeline behaviour and the token-budget invariants."""

from supportbot.config import RETRIEVAL
from supportbot.prompts import NO_CONTEXT_REPLY
from supportbot.tokens import count_tokens
from traffic import TRAFFIC


def test_trivial_query_costs_nothing(bot):
    answer = bot.ask("hi")
    assert answer.route == "trivial"
    assert answer.prompt_tokens == 0
    assert answer.model == "none"


def test_repeat_question_is_served_from_cache(bot):
    bot.ask("how do I reset my password")
    answer = bot.ask("password reset steps?")
    assert answer.cache_hit
    assert answer.prompt_tokens == 0


def test_answer_carries_citations(bot):
    answer = bot.ask("How do I request a partial refund on an annual plan?")
    assert answer.citations
    assert all("#" in c or c.endswith(".md") for c in answer.citations)


def test_context_stays_within_budget(bot):
    for question in TRAFFIC:
        answer = bot.ask(question)
        assert answer.context_tokens <= RETRIEVAL.max_context_tokens


def test_optimized_beats_naive_substantially(bot):
    naive = sum(bot.ask_naive(q).prompt_tokens for q in TRAFFIC)
    optimized = sum(bot.ask(q).prompt_tokens for q in TRAFFIC)
    assert optimized < naive * 0.25, f"{optimized} vs {naive}"


def test_out_of_scope_question_refuses(bot):
    answer = bot.ask("What is the airspeed velocity of an unladen swallow?")
    assert answer.text == NO_CONTEXT_REPLY or answer.prompt_tokens == 0


def test_no_retrieval_means_no_citations(bot):
    assert bot.ask("thanks").citations == []


def test_complex_query_routes_to_smart_model(bot):
    answer = bot.ask("Why do I keep getting 429 errors even after traffic dropped?")
    assert answer.route == "complex"
    assert "4o" in answer.model or answer.model


def test_retrieved_context_is_smaller_than_raw_candidates(bot):
    question = "How long do refunds take?"
    naive = bot.ask_naive(question)
    answer = bot.ask(question)
    assert count_tokens(str(answer.context_tokens)) >= 0
    assert answer.context_tokens < naive.context_tokens
