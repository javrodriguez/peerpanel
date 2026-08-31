"""The headline measurement: planted-error detection, panel vs equal-compute baseline.

Both systems see the same perturbed manuscript, the same corpus, the same
retrieval budget, and — by construction — approximately the same token spend.
The table reports detection per error kind, totals, tokens and wall-clock for
each. Whatever the delta is, it ships.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel

from peerpanel.embeddings import store
from peerpanel.evals.baseline import run_baseline
from peerpanel.evals.planted import detect, plant_errors
from peerpanel.graph.pipeline import chunks_for, embedding_fixture_path
from peerpanel.manuscripts.store import read_manuscript
from peerpanel.orchestration.panel import PanelProviders, run_panel
from peerpanel.providers import OllamaNativeEmbed
from peerpanel.providers.base import ChatProvider
from peerpanel.retrieval import BM25Retriever, Index, VectorRetriever, rrf


class SystemResult(BaseModel):
    system: str
    detected: list[str]  # error_ids
    missed: list[str]
    total_tokens: int
    wall_s: float
    detail: str


class PlantedEvalReport(BaseModel):
    manuscript: str
    n_errors: int
    error_kinds: dict[str, str]  # error_id -> kind
    results: list[SystemResult]
    note: str


NOTE = (
    "Errors are planted in a manuscript held out of the retrieval corpus, so no "
    "unperturbed original is retrievable. Detection requires a finding to NAME the "
    "planted token, which under-credits a described-but-unquoted catch — read the "
    "numbers as a floor. Both systems ran on the same corpus, the same retrieval "
    "budget and approximately the same token spend."
)


def run_planted_eval(
    root: Path,
    manuscript_path: Path,
    providers: PanelProviders,
    baseline_provider: ChatProvider,
    corpus: str = "demo",
) -> PlantedEvalReport:
    header, body = read_manuscript(manuscript_path)
    planted = plant_errors(body, manuscript_path.name)
    planted_path = root / "artifacts" / corpus / f"planted-{manuscript_path.stem}.json"
    planted_path.parent.mkdir(parents=True, exist_ok=True)
    planted_path.write_text(planted.model_dump_json(indent=1) + "\n")

    # The panel reads the perturbed text through a temporary manuscript file so it
    # travels the ordinary production road (header + body), nothing special-cased.
    perturbed_path = root / "artifacts" / corpus / f"perturbed-{manuscript_path.stem}.txt"
    original = manuscript_path.read_text(encoding="utf-8")
    head = original.split("# --- end attribution ---", 1)[0] + "# --- end attribution ---\n"
    perturbed_path.write_text(head + planted.text)

    t0 = time.monotonic()
    review = run_panel(root, perturbed_path, providers, corpus=corpus)
    panel_wall = time.monotonic() - t0
    panel_texts = (
        [f.text for o in review.reviewer_outputs for f in o.findings]
        + [f"{v.verdict} {v.claim_text}" for v in review.verdicts]
        + [f"{d.check} {d.detail}" for d in review.deterministic_findings]
        + [review.summary]
    )

    chunks = chunks_for(root, corpus)
    _chunk_ids, vectors = store.load(embedding_fixture_path(root, corpus))
    index = Index.build(chunks, vectors.astype("float32"))
    chunk_texts = {c.chunk_id: c.text for c in chunks}
    # The baseline gets the strong non-graph rung (RRF hybrid) — the graph is the
    # panel's advantage to demonstrate, not a handicap to impose on the baseline.
    bm25 = BM25Retriever(index)
    vector = VectorRetriever(index, OllamaNativeEmbed())

    def retrieve(query: str) -> list[tuple[str, str]]:
        fused = rrf([bm25.search(query, k=8), vector.search(query, k=8)], k=5)
        return [(h.chunk_id, chunk_texts.get(h.chunk_id, "")) for h in fused]

    t1 = time.monotonic()
    baseline = run_baseline(
        manuscript_text=planted.text,
        retrieve=retrieve,
        title=header.title,
        provider=baseline_provider,
        target_tokens=review.total_tokens,
    )
    baseline_wall = time.monotonic() - t1

    def _result(name: str, texts: list[str], tokens: int, wall: float, detail: str) -> SystemResult:
        hit = [e.error_id for e in planted.errors if detect(e, texts)]
        return SystemResult(
            system=name,
            detected=hit,
            missed=[e.error_id for e in planted.errors if e.error_id not in hit],
            total_tokens=tokens,
            wall_s=round(wall, 1),
            detail=detail,
        )

    return PlantedEvalReport(
        manuscript=manuscript_path.name,
        n_errors=len(planted.errors),
        error_kinds={e.error_id: e.kind for e in planted.errors},
        results=[
            _result(
                "panel", panel_texts, review.total_tokens, panel_wall,
                f"{len(review.reviewer_outputs)} reviewers · {len(review.verdicts)} claims "
                f"verified · {len(review.deterministic_findings)} deterministic findings",
            ),
            _result(
                "single-agent-equal-compute", baseline.finding_texts, baseline.total_tokens,
                baseline_wall, f"{baseline.samples} self-consistency samples",
            ),
        ],
        note=NOTE,
    )


def render_table(report: PlantedEvalReport) -> str:
    lines = [
        f"Planted-error detection · {report.manuscript} · {report.n_errors} errors",
        "",
        f"  {'system':28s} {'detected':>10s} {'tokens':>9s} {'wall_s':>8s}  detail",
    ]
    for result in report.results:
        lines.append(
            f"  {result.system:28s} {len(result.detected):>4d}/{report.n_errors:<5d} "
            f"{result.total_tokens:>9d} {result.wall_s:>8.1f}  {result.detail}"
        )
    lines.append("")
    for result in report.results:
        kinds = [report.error_kinds[e] for e in result.detected]
        lines.append(f"  {result.system} caught: {', '.join(kinds) or '(none)'}")
    lines.append("")
    lines.append(f"  {report.note}")
    return "\n".join(lines)
