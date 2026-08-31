"""GraphRAG global rung: corpus-level questions over community reports.

Two faces, honestly separated:
- retrieve(): deterministic — rank community reports lexically against the
  query, expand the top communities to their members' chunks (Index-filtered).
  This is what the ablation ladder scores.
- answer(): the LLM face — map-reduce over the top-ranked reports (rank
  deterministically, reduce with the model), used by the prior-work reviewer.
  Honest miniaturization of Microsoft's all-report LLM map, and documented as
  such where it is claimed.
"""

from __future__ import annotations

import networkx as nx
from pydantic import BaseModel
from rank_bm25 import BM25Plus

from peerpanel.graph.summaries import CommunityReport
from peerpanel.providers.base import ChatProvider

from .base import Index, RetrievalHit
from .lexical import tokenize

TOP_COMMUNITIES = 8


class GlobalAnswer(BaseModel):
    answer: str
    community_ids: list[int]
    truncated: bool = False


class GraphGlobalRetriever:
    name = "graphrag-global"

    def __init__(
        self,
        index: Index,
        graph: nx.Graph[str],
        reports: list[CommunityReport],
        assignment: dict[str, int],
    ) -> None:
        self._index = index
        self._graph = graph
        self._reports = reports
        self._assignment = assignment
        self._chunk_set = set(index.chunk_ids)
        corpus = [tokenize(f"{r.title} {r.summary} {' '.join(r.member_names)}") for r in reports]
        # BM25Plus: Okapi's IDF hits exactly zero on tiny corpora (df == N/2),
        # and a report set can be small — Plus keeps matching terms positive.
        self._bm25 = BM25Plus(corpus) if corpus else None

    def ranked_reports(self, query: str) -> list[tuple[CommunityReport, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(
            range(len(scores)), key=lambda i: (-scores[i], self._reports[i].community_id)
        )
        return [(self._reports[i], float(scores[i])) for i in order]

    def search(self, query: str, k: int = 10) -> list[RetrievalHit]:
        chunk_scores: dict[str, float] = {}
        top = self.ranked_reports(query)[:TOP_COMMUNITIES]
        for report, community_score in top:
            if community_score <= 0:
                continue
            for node, cid in self._assignment.items():
                if cid != report.community_id or node not in self._graph:
                    continue
                for chunk_id in self._graph.nodes[node]["chunk_ids"]:
                    if chunk_id in self._chunk_set:
                        current = chunk_scores.get(chunk_id, 0.0)
                        chunk_scores[chunk_id] = current + community_score
        ranked = sorted(chunk_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [RetrievalHit(cid, score, rank + 1) for rank, (cid, score) in enumerate(ranked[:k])]

    def answer(
        self, query: str, provider: ChatProvider, top: int = TOP_COMMUNITIES
    ) -> GlobalAnswer:
        ranked = [r for r, s in self.ranked_reports(query)[:top] if s > 0]
        if not ranked:
            return GlobalAnswer(answer="No relevant communities found.", community_ids=[])
        blocks = "\n\n".join(
            f"[community {r.community_id}] {r.title}\n{r.summary}" for r in ranked
        )
        response = provider.chat(
            system=(
                "Answer the question using ONLY the community reports given. Cite the "
                "community ids you drew on in square brackets. If the reports cannot "
                "answer it, say so plainly."
            ),
            user=f"QUESTION: {query}\n\nCOMMUNITY REPORTS:\n{blocks}",
            temperature=0.0,
            max_tokens=1024,
        )
        cited = [r.community_id for r in ranked if f"[community {r.community_id}]" in response.text
                 or f"[{r.community_id}]" in response.text]
        return GlobalAnswer(answer=response.text.strip(), community_ids=cited)
