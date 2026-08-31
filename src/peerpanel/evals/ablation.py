"""The ablation ladder: BM25 → vector → RRF → GraphRAG-local → GraphRAG-global,
run per query case over the exclusion-filtered index, reported honestly.

Per case and rung: the per-item hits table (always) and latency; recall@k /
NDCG@k only when the N gate allows (retrieval_eval.rates_allowed). The losing
rung ships — that is the point.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

import networkx as nx
from pydantic import BaseModel

from peerpanel.embeddings import store
from peerpanel.graph.build import load_graph
from peerpanel.graph.pipeline import chunks_for, embedding_fixture_path
from peerpanel.graph.summaries import CommunityReport
from peerpanel.providers.base import EmbedProvider
from peerpanel.retrieval import (
    BM25Retriever,
    GraphGlobalRetriever,
    GraphLocalRetriever,
    Index,
    VectorRetriever,
    rrf,
)
from peerpanel.retrieval.base import RetrievalHit

from .retrieval_eval import (
    QueryCase,
    build_cases,
    doc_ranking,
    ndcg_at_k,
    per_item_hits,
    rates_allowed,
    recall_at_k,
)

K = 10


class _SearchFn(Protocol):
    def __call__(self, query: str, k: int) -> list[RetrievalHit]: ...


class RungResult(BaseModel):
    rung: str
    case_doi: str
    latency_ms: float
    hits: list[tuple[str, int | None]]  # (relevant doc, rank or None)
    recall_at_k: float | None
    ndcg_at_k: float | None


class AblationReport(BaseModel):
    corpus_manifest: str
    n_cases: int
    skipped_cases: dict[str, str]  # doi -> reason (e.g. twin not a corpus member: no valid run)
    aggregate_relevant: int
    rates_reported: bool
    k: int
    results: list[RungResult]


def _rungs(
    index: Index,
    graph: nx.Graph[str],
    reports: list[CommunityReport],
    assignment: dict[str, int],
    embedder: EmbedProvider,
) -> dict[str, _SearchFn]:
    bm25 = BM25Retriever(index)
    vector = VectorRetriever(index, embedder)
    local = GraphLocalRetriever(index, graph, embedder=embedder)
    global_ = GraphGlobalRetriever(index, graph, reports, assignment)

    def hybrid(query: str, k: int) -> list[RetrievalHit]:
        return rrf([bm25.search(query, k=k * 2), vector.search(query, k=k * 2)], k=k)

    return {
        "bm25": bm25.search,
        "vector": vector.search,
        "rrf-hybrid": hybrid,
        "graphrag-local": local.search,
        "graphrag-global": global_.search,
    }


def load_artifacts(
    root: Path, corpus: str = "ci"
) -> tuple[nx.Graph[str], dict[str, int], list[CommunityReport]]:
    graph = load_graph(root / "artifacts" / corpus / "graph.json")
    communities = json.loads((root / "artifacts" / corpus / "communities.json").read_text())
    assignment = {n: int(c) for n, c in communities["1.0"].items()}
    summaries_path = root / "artifacts" / corpus / "summaries.json"
    reports = [
        CommunityReport.model_validate(r)
        for r in json.loads(summaries_path.read_text())
        if float(r.get("resolution", 0)) == 1.0
    ]
    return graph, assignment, reports


def run_ablation(
    root: Path, embedder: EmbedProvider, k: int = K, corpus: str = "ci"
) -> AblationReport:
    """Run every rung on every case over one corpus's artifacts."""
    manifest_rel = Path("corpus") / ("ci.manifest.json" if corpus == "ci" else "demo.manifest.json")
    all_cases: list[QueryCase] = build_cases(root, manifest_rel)
    from peerpanel.corpus.models import CorpusManifest

    corpus_pmcids = CorpusManifest.load(root / manifest_rel).pmcids()
    # A run is only valid where the manuscript's twin is a member of THIS corpus
    # (the per-RUN reading of the self-exclusion law, DECISIONS D8): a case whose
    # twin is absent has no run here and is skipped LOUDLY, never silently.
    cases = [c for c in all_cases if set(c.excluded_docs) & corpus_pmcids]
    skipped = {
        c.preprint_doi: "twin not a member of this corpus — no valid run"
        for c in all_cases
        if not (set(c.excluded_docs) & corpus_pmcids)
    }
    chunk_ids, vectors = store.load(embedding_fixture_path(root, corpus))
    chunks = chunks_for(root, corpus)
    if [c.chunk_id for c in chunks] != chunk_ids:
        raise RuntimeError("embedding fixture is stale relative to the corpus chunking")
    graph, assignment, reports = load_artifacts(root, corpus)
    allowed = rates_allowed(cases)
    results: list[RungResult] = []
    for case in cases:
        index = Index.build(chunks, vectors.astype("float32"), exclude_docs=set(case.excluded_docs))
        if not index.excluded_docs:
            raise RuntimeError(f"empty exclusion set for {case.preprint_doi} (N3 law)")
        relevant = set(case.relevant_docs)
        for rung_name, search in _rungs(index, graph, reports, assignment, embedder).items():
            t0 = time.monotonic()
            hits = search(case.query, k=k * 3)
            latency_ms = (time.monotonic() - t0) * 1000
            ranking = doc_ranking(hits)
            results.append(
                RungResult(
                    rung=rung_name,
                    case_doi=case.preprint_doi,
                    latency_ms=round(latency_ms, 1),
                    hits=per_item_hits(ranking, relevant, k=k * 3),
                    recall_at_k=round(recall_at_k(ranking, relevant, k), 4) if allowed else None,
                    ndcg_at_k=round(ndcg_at_k(ranking, relevant, k), 4) if allowed else None,
                )
            )
    return AblationReport(
        corpus_manifest=str(manifest_rel),
        n_cases=len(cases),
        skipped_cases=skipped,
        aggregate_relevant=sum(len(c.relevant_docs) for c in cases),
        rates_reported=allowed,
        k=k,
        results=results,
    )


def render_table(report: AblationReport) -> str:
    """Plain-text table: one row per rung, aggregated over cases."""
    lines = [
        f"Ablation over {report.corpus_manifest} · {report.n_cases} cases · "
        f"aggregate relevant {report.aggregate_relevant} · "
        + (
            f"recall@{report.k}/NDCG@{report.k} reported"
            if report.rates_reported
            else "below the N>=20 gate — per-item hits only, no rates"
        )
    ]
    for doi, reason in report.skipped_cases.items():
        lines.append(f"  (skipped {doi}: {reason})")
    rungs: dict[str, list[RungResult]] = {}
    for r in report.results:
        rungs.setdefault(r.rung, []).append(r)
    for rung_name, rows in rungs.items():
        found = sum(1 for row in rows for _, rank in row.hits if rank is not None)
        total = sum(len(row.hits) for row in rows)
        latency = sum(row.latency_ms for row in rows) / len(rows)
        line = f"  {rung_name:16s} hits {found}/{total} · p_mean latency {latency:7.1f}ms"
        if report.rates_reported:
            recall = sum(row.recall_at_k or 0 for row in rows) / len(rows)
            ndcg = sum(row.ndcg_at_k or 0 for row in rows) / len(rows)
            line += f" · recall@{report.k} {recall:.3f} · ndcg@{report.k} {ndcg:.3f}"
        lines.append(line)
    return "\n".join(lines)
