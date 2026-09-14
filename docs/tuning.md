# Tuning guide

Every knob lives in `src/supportbot/config.py`. They are all recall-versus-cost
trades, and the correct setting depends on your corpus and your tolerance for a
wrong answer.

## Method

1. Run `python benchmarks/ablation.py` to see what each lever is currently worth.
2. Change one value.
3. Run `pytest`. The fidelity tests are the floor.
4. Re-run the ablation and compare.

The fidelity tests matter more than the token numbers. Cutting tokens is trivial
if you are allowed to cut the right chunk along with them:

| Test | Protects |
|---|---|
| `test_retrieval_finds_rate_limit_answer` | The answer survives the funnel |
| `test_retrieval_finds_password_reset` | Second topic, same guarantee |
| `test_compression_preserves_document_order` | Procedures stay in order |
| `test_same_topic_different_question_is_rejected` | Cache never answers the wrong question |
| `test_dedupe_keeps_the_higher_scoring_copy` | Dedupe drops the worse duplicate |

Tighten until one fails. That is your floor; back off one step.

## Retrieval

```python
fetch_k = 20        # candidates before filtering
top_k = 3           # chunks that reach the prompt
min_relevance = 0.15
max_context_tokens = 1_200
```

**`fetch_k`** costs nothing at generation time — it is a local vector scan. Raising
it gives the filters more to work with. Lower it only if vector search latency
matters.

**`top_k`** is the dominant token knob. The ablation shows `top_k=10` costs **+33%
tokens** for no measurable quality gain on this corpus. Raise it only when answers
genuinely span documents.

**`min_relevance`** is what lets the bot say "I couldn't find that in the docs"
instead of padding to `top_k` with noise and confabulating from it. Removing it
costs **+11% tokens** and, more importantly, removes the refusal path.

**`max_context_tokens`** is insurance, not a saving. The ablation reports it as
*negligible* because on this corpus it never binds — that is the point. It caps
the damage from a pathological document rather than optimizing the common case.

## Chunking

```python
target_tokens = 220
max_tokens = 400
min_tokens = 40
```

Larger chunks mean fewer, more coherent retrievals but more wasted tokens per
hit. Smaller chunks retrieve precisely but fragment procedures across chunk
boundaries, and you pay to reassemble them.

`min_tokens` controls runt merging. Set it to `0` to disable merging entirely.

## Cache

```python
semantic_hit_threshold = 0.93       # trained embeddings
approximate_hit_threshold = 0.72    # hashing fallback
min_lexical_overlap = 0.5
```

The asymmetry of errors should drive these:

* A **miss** costs one pipeline run — a few hundred tokens.
* A **false hit** gives the customer a confidently wrong answer and you may never
  find out.

Set thresholds high. Inspect `cache.rejected` to see what you are turning away:

```python
for question, matched, score, overlap in bot.cache.rejected:
    print(f"{score:.2f} {overlap:.2f}  {question!r} ~ {matched!r}")
```

If that list is full of genuine paraphrases, lower `min_lexical_overlap` before
lowering the cosine threshold — lexical overlap is the weaker signal and the
safer one to relax.

## Routing

The ablation shows routing saving **0 tokens** but **5 retrievals**. That is not a
failure: the relevance floor already discards greetings, so routing's win is
latency and embedding spend, not generation spend.

Add domain markers to `COMPLEX_MARKERS` to escalate more traffic to the smart
model; add greetings to `GREETINGS`. Note that `GREETINGS` is stemmed at import
time, because `tokenize()` stems its output — a literal `"cheers"` would never
match the stemmed token `cheer`. That bug shipped once and is now pinned by
`test_greetings_skip_retrieval`.

## Switching to a trained embedding

```python
from supportbot.embeddings import get_embeddings
bot = SupportBot(documents, embeddings=get_embeddings(offline=False))
```

Thresholds switch automatically, keyed on the `is_approximate` attribute. Expect
a materially higher cache hit rate: the hashing embedding misses loose
paraphrases such as *"how long until a refund shows up on my card"* versus *"how
long do refunds take"*, which a trained model matches easily.
