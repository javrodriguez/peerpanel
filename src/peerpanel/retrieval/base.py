"""Retrieval core: the filtered Index every mode searches over, and the hit type.

Self-exclusion happens HERE, once, at index construction: excluded documents
(the manuscript-under-review's twin, a planted-error paper's original) never
enter any mode's index, and the Index records what it dropped so a run can
assert its exclusion set is non-empty (the cross-cutting law).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from peerpanel.text.chunks import Chunk

if TYPE_CHECKING:
    import networkx as nx


def doc_of(chunk_id: str) -> str:
    return chunk_id.split(":", 1)[0]


@dataclass(frozen=True)
class RetrievalHit:
    chunk_id: str
    score: float
    rank: int  # 1-based

    @property
    def doc_id(self) -> str:
        return doc_of(self.chunk_id)


@dataclass
class Index:
    chunk_ids: list[str]
    texts: list[str]
    vectors: NDArray[np.float32] | None
    excluded_docs: set[str] = field(default_factory=set)
    dropped_chunk_count: int = 0

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        vectors: NDArray[np.float32] | None = None,
        exclude_docs: frozenset[str] | set[str] = frozenset(),
    ) -> Index:
        if vectors is not None and vectors.shape[0] != len(chunks):
            raise ValueError("one vector row per chunk")
        keep = [i for i, c in enumerate(chunks) if doc_of(c.chunk_id) not in exclude_docs]
        dropped = len(chunks) - len(keep)
        actually_excluded = {
            doc_of(c.chunk_id) for c in chunks if doc_of(c.chunk_id) in exclude_docs
        }
        return cls(
            chunk_ids=[chunks[i].chunk_id for i in keep],
            texts=[chunks[i].text for i in keep],
            vectors=vectors[keep].astype(np.float32) if vectors is not None else None,
            excluded_docs=actually_excluded,
            dropped_chunk_count=dropped,
        )

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def live_nodes(self, graph: nx.Graph[str]) -> set[str]:
        """Graph entities this run may use.

        Chunk-level filtering stops excluded TEXT from being read, but it does
        not stop an excluded document's entities from steering the ranking: an
        entity evidenced only by the twin (its authors, its coined terms) would
        still match a query and vote for its neighbours. Those entities do not
        exist for this run.

        Residual, stated: an entity evidenced by both excluded and surviving
        chunks keeps the edge weight the excluded text contributed. Removing
        that would require rebuilding the graph per run.
        """
        if not self.excluded_docs:
            return set(graph.nodes())
        live: set[str] = set()
        for node, attrs in graph.nodes(data=True):
            evidence = attrs.get("chunk_ids") or set()
            if any(doc_of(cid) not in self.excluded_docs for cid in evidence):
                live.add(node)
        return live
