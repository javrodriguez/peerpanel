"""Prior-work/novelty reviewer — GLOBAL retrieval scope, qwen2 family.

Structural identity: community-level GraphRAG-global retrieval (what does the
corpus as a whole say about this area), contribution-weighted focus, a
different model family from the methods reviewer (heterogeneity is the one
intervention the multi-agent literature endorses without caveat).
"""

from __future__ import annotations

from peerpanel.providers.base import ChatProvider, TokenLedger
from peerpanel.retrieval.graph_global import GraphGlobalRetriever

from .reviewer_base import run_reviewer
from .schemas import ReviewerOutput

REVIEWER_NAME = "prior-work-novelty"
FOCUS = (
    "novelty and contribution relative to the provided literature context — what is "
    "genuinely new here, what overlaps prior work, and whether the framing of the "
    "contribution is supported"
)


def novelty_queries(manuscript_text: str, title: str) -> list[str]:
    lead = " ".join(manuscript_text.split()[:80])
    return [title, lead, "prior work on this mechanism and related pathways"]


def review(
    *,
    manuscript_text: str,
    title: str,
    retriever: GraphGlobalRetriever,
    provider: ChatProvider,
    chunk_texts: dict[str, str],
    ledger: TokenLedger | None = None,
) -> ReviewerOutput:
    def retrieve(query: str) -> list[tuple[str, str]]:
        return [(h.chunk_id, chunk_texts.get(h.chunk_id, "")) for h in retriever.search(query, k=5)]

    return run_reviewer(
        reviewer_name=REVIEWER_NAME,
        provider=provider,
        manuscript_text=manuscript_text,
        retrieve=retrieve,
        focus=FOCUS,
        queries=novelty_queries(manuscript_text, title),
        ledger=ledger,
    )
