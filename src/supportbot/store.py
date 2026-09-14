"""In-memory vector store.

Small enough to read in one sitting, which is the point: the token savings come
from what happens *around* the store, not from the store itself. Swap in FAISS,
pgvector or Chroma without touching the rest of the pipeline.
"""

from __future__ import annotations

from .chunking import Chunk
from .embeddings import cosine


class VectorStore:
    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk]) -> "VectorStore":
        vectors = self.embeddings.embed_documents([c.render() for c in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = vector
        self.chunks.extend(chunks)
        return self

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Return the k nearest chunks with their raw similarity scores."""
        if not self.chunks:
            return []
        query_vector = self.embeddings.embed_query(query)
        scored = [(c, cosine(query_vector, c.embedding)) for c in self.chunks]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self.chunks)
