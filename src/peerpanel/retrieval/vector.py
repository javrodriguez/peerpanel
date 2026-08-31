"""Vector rung — cosine over the embedding fixture (vectors are L2-normalised,
so cosine is a dot product)."""

from __future__ import annotations

import numpy as np

from peerpanel.providers.base import EmbedProvider

from .base import Index, RetrievalHit


class VectorRetriever:
    name = "vector"

    def __init__(self, index: Index, embedder: EmbedProvider) -> None:
        if index.vectors is None:
            raise ValueError("vector retrieval needs an index built with vectors")
        self._index = index
        self._embedder = embedder

    def search(self, query: str, k: int = 10) -> list[RetrievalHit]:
        assert self._index.vectors is not None
        q = self._embedder.embed([query])[0].astype(np.float32)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q = q / norm
        scores = self._index.vectors @ q
        order = sorted(
            range(len(scores)), key=lambda i: (-float(scores[i]), self._index.chunk_ids[i])
        )
        return [
            RetrievalHit(self._index.chunk_ids[i], float(scores[i]), rank + 1)
            for rank, i in enumerate(order[:k])
        ]
