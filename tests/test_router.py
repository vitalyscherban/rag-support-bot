from supportbot.router import Route, canned_reply, classify


def test_greetings_skip_retrieval():
    for greeting in ("hi", "hello", "thanks", "thank you", "cheers", "bye"):
        decision = classify(greeting)
        assert decision.route is Route.TRIVIAL, greeting
        assert not decision.needs_retrieval


def test_factual_lookup_is_simple_and_cheap():
    decision = classify("What is the default API rate limit?")
    assert decision.route is Route.SIMPLE
    assert decision.model_tier == "cheap"


def test_causal_question_escalates_to_smart_model():
    decision = classify("Why do I keep getting 429 errors after traffic dropped?")
    assert decision.route is Route.COMPLEX
    assert decision.model_tier == "smart"


def test_comparison_escalates():
    assert classify("What's the difference between SCIM and SAML?").route is Route.COMPLEX


def test_multipart_question_escalates():
    assert classify("How do I rotate a key? And what is the grace period?").route is Route.COMPLEX


def test_word_containing_greeting_is_not_trivial():
    """'hi' inside 'this' must not trigger the greeting path."""
    assert classify("this endpoint returns a 500 error").route is not Route.TRIVIAL


def test_empty_query_is_trivial():
    assert classify("   ").route is Route.TRIVIAL


def test_canned_replies_are_on_topic():
    assert "docs" in canned_reply("hi").lower() or "help" in canned_reply("hi").lower()
    assert canned_reply("thanks") == "Happy to help!"
