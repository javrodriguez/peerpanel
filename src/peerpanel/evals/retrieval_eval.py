"""Retrieval evaluation: citation-derived ground truth, honestly gated.

Ground truth per query manuscript = the PMC papers it cites (read from its
published twin's reference list, committed in corpus/ground_truth.json) that
are members of the evaluated corpus, minus the manuscript's own twin (the
self-exclusion law). Metrics are doc-level. The N gate is absolute: below 20
aggregate relevant documents, recall@k / NDCG are NOT reported — per-item hit
tables only (a rate on thin N is a decorative number).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import BaseModel

from peerpanel.corpus.models import CorpusManifest
from peerpanel.manuscripts.store import read_manuscript
from peerpanel.manuscripts.twins import load_twins
from peerpanel.retrieval.base import RetrievalHit

N_GATE = 20
QUERY_BODY_WORDS = 150


class QueryCase(BaseModel):
    preprint_doi: str
    manuscript_file: str
    query: str
    relevant_docs: list[str]
    excluded_docs: list[str]


def build_cases(root: Path, corpus_manifest: Path) -> list[QueryCase]:
    ground_truth = json.loads((root / "corpus" / "ground_truth.json").read_text())
    corpus_pmcids = CorpusManifest.load(root / corpus_manifest).pmcids()
    twins = load_twins(root / "manuscripts" / "twins.json")
    cases: list[QueryCase] = []
    for path in sorted((root / "manuscripts").glob("*.txt")):
        header, body = read_manuscript(path)
        entry = ground_truth["manuscripts"].get(header.preprint_doi)
        if entry is None:
            raise KeyError(f"no ground truth recorded for {header.preprint_doi}")
        excluded = twins.excluded_pmcids(header.preprint_doi)
        relevant = (set(entry["cited_pmcids"]) & corpus_pmcids) - excluded
        query = f"{header.title}. " + " ".join(body.split()[:QUERY_BODY_WORDS])
        cases.append(
            QueryCase(
                preprint_doi=header.preprint_doi,
                manuscript_file=path.name,
                query=query,
                relevant_docs=sorted(relevant),
                excluded_docs=sorted(excluded),
            )
        )
    return cases


def rates_allowed(cases: list[QueryCase]) -> bool:
    """The gate: aggregate relevant docs must reach N_GATE for recall/NDCG."""
    return sum(len(c.relevant_docs) for c in cases) >= N_GATE


def doc_ranking(hits: list[RetrievalHit]) -> list[str]:
    """Chunk hits -> ordered unique documents (first hit per doc)."""
    seen: set[str] = set()
    out: list[str] = []
    for hit in hits:
        if hit.doc_id not in seen:
            seen.add(hit.doc_id)
            out.append(hit.doc_id)
    return out


def recall_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranking[:k]) & relevant) / len(relevant)


def ndcg_at_k(ranking: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance NDCG@k."""
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2) for i, doc in enumerate(ranking[:k]) if doc in relevant
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def per_item_hits(ranking: list[str], relevant: set[str], k: int) -> list[tuple[str, int | None]]:
    """The always-reported table row source: (relevant doc, its rank or None)."""
    position = {doc: i + 1 for i, doc in enumerate(ranking[:k])}
    return [(doc, position.get(doc)) for doc in sorted(relevant)]
