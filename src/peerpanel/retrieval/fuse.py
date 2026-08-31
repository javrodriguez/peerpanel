"""Reciprocal-rank fusion — the hybrid rung (RRF, k=60 per the literature)."""

from __future__ import annotations

from .base import RetrievalHit

RRF_K = 60


def rrf(hit_lists: list[list[RetrievalHit]], k: int = 10, rrf_k: int = RRF_K) -> list[RetrievalHit]:
    scores: dict[str, float] = {}
    for hits in hit_lists:
        for hit in hits:
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + hit.rank)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [RetrievalHit(cid, score, rank + 1) for rank, (cid, score) in enumerate(ranked[:k])]
