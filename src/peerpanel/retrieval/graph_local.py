"""GraphRAG local rung: entity-anchored retrieval.

Query terms match graph entities (deterministic normalisation); matched
entities and their weighted neighbours vote for the chunks they came from;
an optional embedding blend refines the ranking. Only chunks present in the
(already exclusion-filtered) Index can surface.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from peerpanel.graph.extract import candidate_terms
from peerpanel.providers.base import EmbedProvider

from .base import Index, RetrievalHit

NEIGHBOUR_DAMP = 0.5
COSINE_BLEND = 1.0


def _norm(name: str) -> str:
    return " ".join(name.split()).lower()


class GraphLocalRetriever:
    name = "graphrag-local"

    def __init__(
        self,
        index: Index,
        graph: nx.Graph[str],
        embedder: EmbedProvider | None = None,
    ) -> None:
        self._index = index
        self._graph = graph
        self._embedder = embedder if index.vectors is not None else None
        self._position = {cid: i for i, cid in enumerate(index.chunk_ids)}

    def _matched_nodes(self, query: str) -> dict[str, float]:
        terms = [_norm(t) for t in candidate_terms(query)]
        terms += [t for t in _norm(query).split() if len(t) >= 4]
        weights: dict[str, float] = {}
        for node in self._graph.nodes():
            for term in terms:
                if node == term or (len(term) >= 4 and term in node):
                    weights[node] = max(weights.get(node, 0.0), 1.0)
                    break
        for seed in list(weights):
            edges = self._graph[seed]
            if not edges:
                continue
            max_w = max(attrs["weight"] for attrs in edges.values())
            for neighbour, attrs in edges.items():
                bonus = NEIGHBOUR_DAMP * attrs["weight"] / max_w
                weights[neighbour] = max(weights.get(neighbour, 0.0), bonus)
        return weights

    def search(self, query: str, k: int = 10) -> list[RetrievalHit]:
        node_weights = self._matched_nodes(query)
        chunk_scores: dict[str, float] = {}
        for node, weight in node_weights.items():
            for chunk_id in self._graph.nodes[node]["chunk_ids"]:
                if chunk_id in self._position:
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight
        if self._embedder is not None and chunk_scores:
            assert self._index.vectors is not None
            q: NDArray[np.float32] = self._embedder.embed([query])[0].astype(np.float32)
            norm = float(np.linalg.norm(q))
            if norm > 0:
                q = q / norm
            for chunk_id in chunk_scores:
                cosine = float(self._index.vectors[self._position[chunk_id]] @ q)
                chunk_scores[chunk_id] += COSINE_BLEND * cosine
        ranked = sorted(chunk_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [RetrievalHit(cid, score, rank + 1) for rank, (cid, score) in enumerate(ranked[:k])]
