# Architecture

## Design premise

Every stage in this pipeline is cheaper than the stage after it. Traffic is
filtered by the cheap stages so the expensive one — generation — only ever sees
what genuinely needs it, and sees as little of it as possible.

```mermaid
flowchart TD
    Q["Incoming question"] --> R{"1 · Router<br/><i>rules, 0 tokens</i>"}

    R -->|trivial| CANNED["Canned reply<br/><b>0 tokens, 0 retrievals</b>"]
    R -->|simple / complex| C{"2 · Semantic cache<br/><i>1 embedding</i>"}

    C -->|hit| CACHED["Cached answer<br/><b>0 prompt tokens</b>"]
    C -->|miss| S["3 · Vector search<br/><i>fetch_k = 20</i>"]

    S --> F["4 · Relevance floor<br/><i>drop score &lt; 0.15</i>"]
    F --> D["5 · Dedupe<br/><i>drop cosine ≥ threshold</i>"]
    D --> M["6 · MMR select<br/><i>top_k = 3</i>"]
    M --> X["7 · Sentence compression<br/><i>4 sentences per chunk</i>"]
    X --> CEIL["8 · Context ceiling<br/><i>≤ 1,200 tokens</i>"]
    CEIL --> G["9 · Generate<br/><b>the only paid stage</b>"]

    G --> STORE["Store in cache"]
    STORE --> A["Answer + citations"]
    CANNED --> A
    CACHED --> A

    classDef free fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
    classDef cheap fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
    classDef paid fill:#fff3e0,stroke:#ef6c00,color:#e65100
    class R,F,D,M,X,CEIL,CANNED free
    class C,S,STORE cheap
    class G paid
```

Green stages cost nothing, blue stages cost an embedding, orange is the only
stage billed at generation rates.

## Where the tokens go

Naive RAG sends 20 raw chunks. Each filter removes a different *class* of waste,
which is why they compose rather than overlap completely.

```mermaid
flowchart LR
    A["20 chunks<br/>~1,580 tokens"] -->|relevance floor| B["~8 chunks<br/>weak matches gone"]
    B -->|dedupe| C["~6 chunks<br/>FAQ restatements gone"]
    C -->|MMR top_k=3| D["3 chunks<br/>diverse facets"]
    D -->|sentence compression| E["3 chunks<br/>~190 tokens"]

    style A fill:#ffcdd2,stroke:#c62828,color:#b71c1c
    style E fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20
```

## Request lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant R as Router
    participant SC as Semantic cache
    participant VS as Vector store
    participant RR as Rerank + compress
    participant LLM as Model

    U->>R: "how do I reset my password"
    R->>R: classify (rules, free)

    alt trivial (greeting)
        R-->>U: canned reply · 0 tokens
    else needs documents
        R->>SC: lookup(question)
        alt semantic hit
            SC-->>U: cached answer · 0 prompt tokens
        else miss
            SC->>VS: search(k=20)
            VS-->>RR: 20 candidates
            RR->>RR: floor → dedupe → MMR → compress
            RR->>LLM: system + 3 compressed chunks + question
            LLM-->>SC: answer
            SC->>SC: store(question, answer)
            SC-->>U: answer + citations
        end
    end
```

## Ingestion

Chunk quality decides how few chunks retrieval can get away with. Fixed-size
chunking forces you to retrieve more to compensate; structure-aware chunking
with heading breadcrumbs makes a single chunk self-describing.

```mermaid
flowchart TD
    MD["Markdown docs"] --> SEC["Split on headings<br/><i>fences treated as opaque</i>"]
    SEC --> PATH["Attach heading path<br/><i>Billing &gt; Refunds &gt; Partial</i>"]
    PATH --> OVER{"over max_tokens?"}
    OVER -->|yes| SPLIT["Split: paragraph → sentence → word"]
    OVER -->|no| RUNT{"under min_tokens?"}
    SPLIT --> RUNT
    RUNT -->|yes| MERGE["Merge forward one step<br/><i>keep child breadcrumb</i>"]
    RUNT -->|no| EMB["Embed"]
    MERGE --> EMB
    EMB --> IDX[("Vector index")]
```

### Why merge forward, and only once

A stub section (`## Overview` with one sentence) is noise alone but useful glued
to what it introduces. Two rules keep that honest:

* **Forward, not backward** — the following chunk's heading path is the more
  specific of the two. Merging backward would relabel a nested section with its
  parent's breadcrumb.
* **One step only** — without a cap, a document whose sections are all short
  cascades into a single chunk carrying the deepest breadcrumb, a label that is
  false for most of its content. `test_all_short_document_does_not_collapse_to_one_chunk`
  pins this.

## The two-signal cache gate

The semantic cache is the highest-yield lever and the most dangerous one. A false
hit answers the customer's question with someone else's answer, confidently.

```mermaid
flowchart TD
    Q["New question"] --> E["Embed"]
    E --> N["Nearest cached entry"]
    N --> T{"cosine ≥ threshold?"}
    T -->|no| MISS["Miss → full pipeline"]
    T -->|yes| L{"lexical overlap ≥ 0.5?"}
    L -->|no| REJ["Reject + record for tuning"]
    L -->|yes| HIT["Hit · 0 prompt tokens"]
    REJ --> MISS

    style HIT fill:#c8e6c9,stroke:#2e7d32,color:#1b5e20
    style REJ fill:#ffe0b2,stroke:#ef6c00,color:#e65100
    style MISS fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
```

Embeddings alone rate *"what is the default rate limit"* and *"how do I raise my
rate limit"* at **0.577** cosine — the same topic with different answers — while a
genuine paraphrase pair scores **0.408**. No single threshold separates them, so
a second, independent signal is required. Requiring lexical agreement too trades
hit rate for safety, which is the correct direction when the failure is silent.

## Module map

| Module | Lever | Cost |
|---|---|---|
| `router.py` | Skip retrieval; pick model tier | free (rules) |
| `cache.py` | Semantic cache, two-signal gate | 1 embedding |
| `chunking.py` | Structure-aware, self-describing chunks | ingest-time |
| `store.py` | Vector search | 1 embedding |
| `rerank.py` | Relevance floor, dedupe, MMR | vector math |
| `compress.py` | Sentence extraction, context ceiling | vector math |
| `pipeline.py` | Orchestration and accounting | — |

## Calibration is backend-specific

Thresholds are a property of the embedding model, not of the algorithm. This
repo ships a deterministic hashing embedding so everything runs offline, and its
similarity distribution is nothing like a trained model's:

| Threshold | Trained model | Hashing fallback |
|---|---|---|
| Cache hit | 0.93 | 0.72 |
| Dedupe | 0.92 | 0.70 |

Reusing the trained-model numbers offline silently turns dedupe into a no-op and
drives the cache hit rate to zero — failures that look like "the feature does
nothing" rather than an error. `RetrievalBudget.dedupe_for()` and
`SemanticCache.__init__` select by backend, and
`test_dedupe_threshold_is_backend_aware` pins it.
