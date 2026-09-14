# Evaluation

Token savings are easy to fake by degrading answers. These are the checks that
keep the numbers honest.

## Offline, no API key

```bash
python benchmarks/compare.py     # naive vs optimized on realistic traffic
python benchmarks/ablation.py    # what each lever is individually worth
pytest                           # 43 tests, fidelity + efficiency
```

All three run on the standard library plus `tiktoken`. The deterministic hashing
embedding means the numbers are reproducible run to run.

## Traffic shape matters more than volume

`benchmarks/traffic.py` is roughly 20% trivial, 45% reworded repeats, 35% novel.
Benchmarking on 50 unique hard questions would flatter the naive baseline and
hide the two levers that pay most in production — routing and caching. If your
real distribution differs, replace `TRAFFIC` before trusting any number here.

## What each number means

| Metric | Source | Meaning |
|---|---|---|
| prompt tokens | `Answer.prompt_tokens` | Billed at generation rates |
| retrievals | ablation | Embedding + vector search round-trips |
| cache hit rate | `bot.cache.stats.hit_rate` | Fraction served with zero prompt tokens |
| savings % | `Answer.savings_pct` | Versus top-20-raw on the same question |

## Measuring in production

Set `LANGCHAIN_TRACING_V2=true` and read per-run prompt tokens from LangSmith.
`Answer` also carries its own accounting, so you can assert on it in CI and fail
the build when a change regresses the budget:

```python
answer = bot.ask(question)
assert answer.context_tokens <= RETRIEVAL.max_context_tokens
```

## Answer quality

Token efficiency without a quality gate is just truncation. The retrieval
fidelity tests are the minimum bar. For a real deployment, add:

1. **A golden set.** 30–50 real questions with known-correct source documents.
   Assert the right document appears in `answer.citations`. This catches funnel
   regressions that token counts never will.
2. **Refusal accuracy.** Out-of-scope questions must produce
   `"I couldn't find that in the docs"` rather than a confident guess. Test both
   directions — over-refusal is also a failure.
3. **Cache audit.** Periodically sample cache hits and confirm the served answer
   still fits the new question. `cache.rejected` shows the near-misses you turned
   away; the hits are what need review.

## Known limitations

* The hashing embedding is a bag of words. It matches paraphrases that share
  vocabulary and misses those that do not, which caps the offline cache hit rate
  at around 15%. A trained embedding does materially better.
* The corpus is four documents. Dedupe and MMR both get more valuable as a corpus
  grows and accumulates restatements.
* Sentence compression uses embedding similarity. `LLMChainExtractor` is more
  accurate but spends a model call per chunk — worth it only when chunks are long
  and traffic is low.
