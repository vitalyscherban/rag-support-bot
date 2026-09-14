"""A realistic support traffic mix.

The shape matters more than the volume. Real queues are dominated by repeats and
pleasantries; benchmarking on 50 unique hard questions flatters the naive
baseline and hides the two levers (routing, semantic caching) that pay most.

Roughly: 20% trivial, 45% repeats of an earlier question in different words,
35% genuinely novel.
"""

TRAFFIC: list[str] = [
    # -- novel questions -------------------------------------------------
    "How do I request a partial refund on an annual plan?",
    "hi",
    "What is the default API rate limit?",
    "how do i reset my password",
    "Why do I keep getting 429 errors even after traffic dropped?",
    # -- repeats, reworded -----------------------------------------------
    "i forgot my password, how do i reset it?",
    "thanks",
    "can i get a partial refund for my yearly subscription?",
    "What's the rate limit per API key by default?",
    "How long do refunds take to arrive?",
    "hello",
    "password reset steps?",
    "how long until a refund shows up on my card",
    # -- novel again -------------------------------------------------------
    "Can I rotate an API key without downtime?",
    "What happens after a payment fails three times?",
    "thank you",
    "How do I stop SSO from locking everyone out?",
    "what is the partial refund policy",
    "Why didn't my password reset email arrive?",
    "How do I raise my rate limit?",
    "cheers",
    "is there a grace period when rotating keys?",
    "What's the difference between SCIM deprovisioning and deleting a user?",
    "How many requests per minute can one key handle?",
    "how do i get a refund",
]


def traffic_profile() -> dict[str, int]:
    """Used by the README to describe the mix without hand-counting it."""
    from supportbot.router import Route, classify

    counts = {route.value: 0 for route in Route}
    for question in TRAFFIC:
        counts[classify(question).route.value] += 1
    return counts
