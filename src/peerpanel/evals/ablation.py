"""The ablation ladder: BM25 → vector → RRF → GraphRAG-local → GraphRAG-global,
run per query case over the exclusion-filtered index, reported honestly.

Per case and rung: the per-item hits table (always) and latency; recall@k /
NDCG@k only when the N gate allows (retrieval_eval.rates_allowed). The losing
rung ships — that is the point.

`RUN_VARYING_FIELDS` names every field of this schema whose value is expected to
differ between two runs over identical bytes: `latency_ms` (a wall-clock
measurement) and `run_utc` (when the run happened). Everything else is a
deterministic function of the committed corpus, extractions and query vectors, so
a difference anywhere else means the record was not produced by this code. The
list lives here, beside the models, because the conformance test imports it
rather than repeating it — a field that becomes run-varying is exempted in one
place, and a field that stops being run-varying cannot stay quietly exempt.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import networkx as nx
from pydantic import BaseModel

from peerpanel.embeddings import store
from peerpanel.graph.build import load_graph
from peerpanel.graph.pipeline import chunks_for, embedding_fixture_path
from peerpanel.graph.run_graph import build_run_graph
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
    recall_ceiling_at_k,
)

K = 10
DOC_DEPTH_MULTIPLIER = 3  # the deeper list, in DOCUMENTS
MAX_CHUNKS = 2000

# Declared run-varying, at both levels of the schema: `latency_ms` on a rung row and
# `run_utc` on the report. Anything else that moves between two runs over the same
# committed bytes is a defect, not a timestamp.
RUN_VARYING_FIELDS = ("latency_ms", "run_utc")


class _SearchFn(Protocol):
    def __call__(self, query: str, k: int) -> list[RetrievalHit]: ...


class RungResult(BaseModel):
    rung: str
    case_doi: str
    latency_ms: float
    chunks_retrieved: int  # how deep the CHUNK search had to go for equal doc depth
    docs_ranked: int  # the document list actually scored
    hits: list[tuple[str, int | None]]  # (relevant doc, rank or None)
    recall_at_k: float | None
    recall_ceiling_at_k: float | None  # the best recall@k achievable for this case
    ndcg_at_k: float | None


class AblationReport(BaseModel):
    """One ladder run.

    `run_utc` is REQUIRED and has no default: a record that cannot say when it was
    produced cannot show that a regeneration actually happened, and a default would
    let a record carrying no timestamp load as though it carried one (D18 — a field
    whose absence changes a number's meaning has no default). It is written as
    ISO-8601 UTC to the second, and it is declared run-varying in
    `RUN_VARYING_FIELDS` alongside `latency_ms`; those two are the only fields a
    regeneration over identical bytes may move.
    """

    corpus_manifest: str
    run_utc: str  # ISO-8601 UTC, seconds — run-varying by declaration
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
    """Graph, community assignment and reports for a corpus.

    Derived artifacts under `artifacts/` are gitignored because they are bulky
    and regenerable, so a clean clone has none of them. Everything they contain
    is reconstructed here from the COMMITTED caches — the same road quickstart
    takes — which is what lets a stranger reproduce every retrieval number with
    no model and no prior run.
    """
    graph_path = root / "artifacts" / corpus / "graph.json"
    communities_path = root / "artifacts" / corpus / "communities.json"
    if graph_path.exists() and communities_path.exists():
        graph = load_graph(graph_path)
        assignment = {n: int(c) for n, c in json.loads(communities_path.read_text())["1.0"].items()}
    else:
        graph, assignment, _withheld = build_run_graph(root, corpus, set())
    summaries_path = root / "artifacts" / corpus / "summaries.json"
    if summaries_path.exists():
        raw = json.loads(summaries_path.read_text())
    else:
        raw = _reports_from_cache(root, corpus, graph, assignment)
    reports = [
        CommunityReport.model_validate(r) for r in raw if float(r.get("resolution", 0)) == 1.0
    ]
    return graph, assignment, reports


def _reports_from_cache(
    root: Path, corpus: str, graph: nx.Graph[str], assignment: dict[str, int]
) -> list[dict[str, object]]:
    """Community reports from the committed summary cache — never a model call."""
    from peerpanel.graph.run_graph import _CacheOnly
    from peerpanel.graph.summaries import SUMMARY_PROVIDER_NAME, summarise_communities

    reports = summarise_communities(
        graph,
        assignment,
        1.0,
        _CacheOnly(SUMMARY_PROVIDER_NAME),
        cache_dir=root / "fixtures" / "summaries" / corpus,
    )
    return [r.model_dump() for r in reports]


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
    _full_graph, assignment, reports = load_artifacts(root, corpus)
    allowed = rates_allowed(cases)
    results: list[RungResult] = []
    for case in cases:
        index = Index.build(chunks, vectors.astype("float32"), exclude_docs=set(case.excluded_docs))
        if not index.dropped_chunk_count:
            raise RuntimeError(f"exclusion removed no chunks for {case.preprint_doi} (N3 law)")
        # Exact per-run graph: the excluded document's extractions are withheld from
        # the merge, so its edge-weight contribution disappears too (D10).
        graph, _run_assignment, _withheld = build_run_graph(root, corpus, set(case.excluded_docs))
        relevant = set(case.relevant_docs)
        for rung_name, search in _rungs(index, graph, reports, assignment, embedder).items():
            # EQUAL DOCUMENT DEPTH. Retrieving a fixed number of CHUNKS and then
            # scoring a DOCUMENT ranking compares lists of wildly different length
            # — measured: 4 documents for RRF against 13 for graph-local on the same
            # query, both reported as "recall@10". Chunk depth is therefore grown
            # until every rung offers the same number of documents.
            want_docs = k * DOC_DEPTH_MULTIPLIER
            chunk_k = want_docs * 4
            ranking: list[str] = []
            while True:
                ranking = doc_ranking(search(case.query, k=chunk_k))
                if len(ranking) >= want_docs or chunk_k >= MAX_CHUNKS:
                    break
                chunk_k *= 2
            # Timed separately, and warm: spaCy loads lazily inside the graph rungs,
            # so the first timed call charges that one-off import to the rung
            # (measured 912ms then 101ms for the same work).
            search(case.query, k=chunk_k)
            t0 = time.monotonic()
            search(case.query, k=chunk_k)
            latency_ms = (time.monotonic() - t0) * 1000
            ranking = ranking[:want_docs]
            results.append(
                RungResult(
                    rung=rung_name,
                    case_doi=case.preprint_doi,
                    latency_ms=round(latency_ms, 1),
                    chunks_retrieved=chunk_k,
                    docs_ranked=len(ranking),
                    hits=per_item_hits(ranking, relevant, k=want_docs),
                    recall_at_k=round(recall_at_k(ranking, relevant, k), 4) if allowed else None,
                    recall_ceiling_at_k=(
                        round(recall_ceiling_at_k(relevant, k), 4) if allowed else None
                    ),
                    ndcg_at_k=round(ndcg_at_k(ranking, relevant, k), 4) if allowed else None,
                )
            )
    return AblationReport(
        corpus_manifest=str(manifest_rel),
        # Stamped at the END of the run, so the value names a run that completed.
        run_utc=datetime.now(UTC).isoformat(timespec="seconds"),
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
            f"recall@{report.k} (shown against its achievable ceiling) / NDCG@{report.k}"
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
        # Two counts, both labelled. found@k is the one the reported rates use;
        # the deeper count says where the rest landed. Publishing only the deeper
        # number under a k-headed table would flatter whichever rung has a long
        # tail — which is exactly the rung whose ranking is worst.
        deep_k = report.k * 3
        found_at_k = sum(
            1 for row in rows for _, rank in row.hits if rank is not None and rank <= report.k
        )
        found_deep = sum(1 for row in rows for _, rank in row.hits if rank is not None)
        total = sum(len(row.hits) for row in rows)
        latency = sum(row.latency_ms for row in rows) / len(rows)
        line = (
            f"  {rung_name:16s} found@{report.k} {found_at_k}/{total} · "
            f"found@{deep_k} {found_deep}/{total} · mean latency {latency:7.1f}ms"
        )
        if report.rates_reported:
            recall = sum(row.recall_at_k or 0 for row in rows) / len(rows)
            ceiling = sum(row.recall_ceiling_at_k or 0 for row in rows) / len(rows)
            ndcg = sum(row.ndcg_at_k or 0 for row in rows) / len(rows)
            # Macro-average the SHARE, like every other rate in this table — a
            # ratio-of-means silently reweights toward the case with the larger
            # denominator, and measured here it flipped which rung appeared to win.
            shares = [
                (row.recall_at_k or 0) / row.recall_ceiling_at_k
                for row in rows
                if row.recall_ceiling_at_k
            ]
            share = sum(shares) / len(shares) if shares else 0.0
            line += (
                f" · recall@{report.k} {recall:.3f}/{ceiling:.3f} ceiling ({share:.0%}) "
                f"· ndcg@{report.k} {ndcg:.3f}"
            )
        lines.append(line)
    # Per-case rows are NOT optional. A mean over a handful of cases can invert the
    # ordering the mean implies — measured here — so the aggregate is never shown
    # without the numbers it was computed from.
    lines.append("")
    lines.append(f"  per case (n={report.n_cases} — read these before trusting any mean):")
    for rung_name, rows in rungs.items():
        for row in rows:
            case = row.case_doi.split("/")[-1]
            rates = (
                f"recall {row.recall_at_k} · ndcg {row.ndcg_at_k}"
                if row.recall_at_k is not None
                else "rates withheld (below the N gate)"
            )
            # Latency belongs here too: its per-case spread is the widest of any
            # column (measured 913ms vs 101ms on one rung), so a mean hides more
            # here than anywhere else — which is exactly what D14 is about.
            lines.append(f"    {rung_name:16s} {case:22s} {rates} · {row.latency_ms:.1f}ms")
    return "\n".join(lines)
