"""BM25 rung — the lexical baseline every fancier mode must beat honestly."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from .base import Index, RetrievalHit

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    name = "bm25"

    def __init__(self, index: Index) -> None:
        self._index = index
        self._bm25 = BM25Okapi([tokenize(t) for t in index.texts])

    def search(self, query: str, k: int = 10) -> list[RetrievalHit]:
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self._index.chunk_ids[i]))
        return [
            RetrievalHit(self._index.chunk_ids[i], float(scores[i]), rank + 1)
            for rank, i in enumerate(order[:k])
        ]
