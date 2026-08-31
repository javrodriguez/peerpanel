"""Methods/statistics reviewer — LOCAL retrieval scope, llama3.1 family.

Structural identity: entity-level GraphRAG-local retrieval (what specific
methods, genes and measurements does the literature pair with this work),
soundness-weighted focus.
"""

from __future__ import annotations

import re

from peerpanel.providers.base import ChatProvider, TokenLedger
from peerpanel.retrieval.graph_local import GraphLocalRetriever

from .reviewer_base import run_reviewer
from .schemas import ReviewerOutput

REVIEWER_NAME = "methods-statistics"
FOCUS = (
    "methodological and statistical soundness — experimental design, controls, "
    "replication, quantification, whether stated methods support the stated conclusions"
)

_METHODS_HEADING = re.compile(
    r"^\s*(methods|materials and methods|experimental)", re.IGNORECASE | re.MULTILINE
)


def methods_queries(manuscript_text: str, title: str) -> list[str]:
    """Deterministic query derivation: the title, and a methods-section lead when found."""
    queries = [title]
    match = _METHODS_HEADING.search(manuscript_text)
    if match:
        tail = manuscript_text[match.start() :]
        queries.append(" ".join(tail.split()[:60]))
    queries.append("statistical analysis quantification replication controls")
    return queries


def review(
    *,
    manuscript_text: str,
    title: str,
    retriever: GraphLocalRetriever,
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
        queries=methods_queries(manuscript_text, title),
        ledger=ledger,
    )
