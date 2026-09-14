"""Chunking decides what retrieval can ever find, so it gets pinned hard."""

from supportbot.chunking import chunk_corpus, chunk_markdown
from supportbot.config import CHUNKS
from supportbot.tokens import count_tokens

SAMPLE = """# Guide

Intro text that is deliberately long enough to stand as its own chunk rather
than being folded into the section that follows it, because the merge rules
only apply to genuinely tiny fragments below the configured floor value here.

## Section A

Body of A with enough words to matter for the retrieval stage, padded out so
that it clears the minimum chunk size and survives as an independent unit in
the index rather than being merged into a neighbouring section during chunking.

### Nested A1

Details about A1, again long enough to clear the runt threshold so the nested
heading path is preserved end to end through chunking and into the breadcrumb
that the retriever will later show to the model as a citation label for it.

## Section B

```python
def example():
    # a heading-looking line inside a fence
    ## not a heading
    return 1
```

Trailing prose after the fence that keeps this section above the minimum size
so the fenced code block is not merged away into some other unrelated chunk.
"""


def test_heading_path_is_tracked():
    chunks = chunk_markdown(SAMPLE, "guide.md")
    paths = {c.breadcrumb for c in chunks}
    assert "Guide > Section A" in paths
    assert "Guide > Section A > Nested A1" in paths


def test_fenced_code_is_not_split_on_inner_hashes():
    """A '## not a heading' inside a fence must not start a new section."""
    chunks = chunk_markdown(SAMPLE, "guide.md")
    code_chunks = [c for c in chunks if "def example()" in c.text]
    assert len(code_chunks) == 1
    assert "## not a heading" in code_chunks[0].text
    assert "return 1" in code_chunks[0].text


def test_tiny_sections_merge_forward_keeping_the_specific_breadcrumb():
    """A stub intro folds into the section it introduces, not the other way round."""
    chunks = chunk_markdown(
        "# T\n\nStub.\n\n## Real section\n\n" + ("detail " * 80), "t.md"
    )
    assert len(chunks) == 1
    assert chunks[0].breadcrumb == "T > Real section"
    assert chunks[0].text.startswith("Stub.")


def test_all_short_document_does_not_collapse_to_one_chunk():
    """Regression: an unbounded merge cascade produced one chunk with a false label."""
    markdown = "# D\n\nOne.\n\n## A\n\nTwo.\n\n## B\n\nThree.\n\n## C\n\nFour.\n"
    chunks = chunk_markdown(markdown, "d.md")
    assert len(chunks) > 1
    assert all(c.breadcrumb.startswith("D") for c in chunks)


def test_chunks_are_self_describing():
    """Rendered chunks carry their breadcrumb, which is what lets top_k stay at 3."""
    chunk = chunk_markdown(SAMPLE, "guide.md")[1]
    assert chunk.render().startswith("[guide.md#")


def test_chunks_respect_max_tokens():
    for chunk in chunk_corpus({"big.md": "# T\n\n" + ("word " * 3000)}):
        assert count_tokens(chunk.text) <= CHUNKS.max_tokens * 1.3


def test_real_corpus_chunks_have_citations(documents):
    chunks = chunk_corpus(documents)
    assert len(chunks) > 10
    assert all(c.citation for c in chunks)
    assert all(c.chunk_id for c in chunks)


def test_ids_are_unique(documents):
    chunks = chunk_corpus(documents)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
