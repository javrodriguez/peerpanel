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

from peerpanel.agents.reviewer_base import EXCERPT_WORDS, excerpt
from peerpanel.corpus.models import CorpusManifest
from peerpanel.embeddings import store
from peerpanel.evals.baseline import run_baseline
from peerpanel.evals.planted import detect, plant_errors
from peerpanel.graph.pipeline import chunks_for, embedding_fixture_path
from peerpanel.manuscripts.store import read_manuscript
from peerpanel.manuscripts.twins import load_twins
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
    excluded_docs: list[str]  # what this arm's index withheld
    dropped_chunks: int  # and how many chunks that actually removed


class PlantedEvalReport(BaseModel):
    manuscript: str
    twin_in_corpus: bool  # was there anything to exclude at all?
    n_errors: int
    error_kinds: dict[str, str]  # error_id -> kind
    results: list[SystemResult]
    note: str


SCORING_NOTE = (
    "Scored over ASSERTION channels only — reviewer findings, REFUTES verdicts and "
    "deterministic-lens findings for the panel; model-authored findings for the "
    "baseline. Restatements of the manuscript and the converger's prose are excluded: "
    "counting them would credit the panel for quoting a planted error, or for agreeing "
    "with it. Detection requires a finding to NAME the planted token, which "
    "under-credits a described-but-unquoted catch — read the numbers as a floor."
)


def _budget_note(panel_tokens: int, baseline_tokens: int, twin_in_corpus: bool) -> str:
    """The note is DERIVED, never asserted: an earlier hardcoded version claimed a
    matched budget the run's own numbers contradicted."""
    share = baseline_tokens / panel_tokens if panel_tokens else 0.0
    budget = (
        f"Token spend: panel {panel_tokens}, baseline {baseline_tokens} "
        f"({share:.0%} of the panel's)."
    )
    if share < 0.9:
        budget += (
            " The baseline hit its sampling ceiling before matching the panel, so this "
            "is NOT an equal-compute comparison — the baseline had less."
        )
    exclusion = (
        "Both arms searched the same index with the same exclusions."
        if twin_in_corpus
        else "The subject is held out of this corpus entirely, so neither arm had "
        "anything to withhold and no answer key existed for either."
    )
    return f"{SCORING_NOTE} {exclusion} {budget}"


def run_planted_eval(
    root: Path,
    manuscript_path: Path,
    providers: PanelProviders,
    baseline_provider: ChatProvider,
    corpus: str = "demo",
) -> PlantedEvalReport:
    header, body = read_manuscript(manuscript_path)
    excluded = load_twins(root / "manuscripts" / "twins.json").excluded_pmcids(
        header.preprint_doi
    )
    # Plant only where the reviewers actually look: both arms read excerpt(body).
    planted = plant_errors(body, manuscript_path.name, window_words=EXCERPT_WORDS)
    visible = excerpt(planted.text)
    unreachable = [e.error_id for e in planted.errors if e.detection_token not in visible]
    if unreachable:
        raise RuntimeError(
            f"planted errors outside the reviewed window: {unreachable} — neither arm "
            "could see them, so the measurement could not be earned"
        )
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
    # ASSERTION CHANNELS ONLY. A claim's text is a near-verbatim slice of the
    # manuscript, so counting it would credit the panel for RESTATING a planted
    # error — and a SUPPORTS verdict would score as a catch for agreeing with it.
    # The converger's prose is excluded for the same reason: it quotes findings
    # rather than asserting new ones, and the baseline has no equivalent channel,
    # so including either would hand the panel surface area the baseline lacks.
    # Only a REFUTES verdict is an assertion that something is wrong.
    panel_texts = (
        [f.text for o in review.reviewer_outputs for f in o.findings]
        + [v.claim_text for v in review.verdicts if v.verdict == "REFUTES"]
        + [f"{d.check} {d.detail}" for d in review.deterministic_findings]
    )

    chunks = chunks_for(root, corpus)
    _chunk_ids, vectors = store.load(embedding_fixture_path(root, corpus))
    # The baseline MUST withhold exactly what the panel withheld. Handing one arm the
    # manuscript's published twin — which states every planted fact correctly — would
    # not be an equal-compute comparison, it would be an answer key.
    index = Index.build(chunks, vectors.astype("float32"), exclude_docs=excluded)
    manifest_rel = "ci.manifest.json" if corpus == "ci" else "demo.manifest.json"
    twin_present = bool(
        excluded & CorpusManifest.load(root / "corpus" / manifest_rel).pmcids()
    )
    if twin_present and not index.dropped_chunk_count:
        raise RuntimeError(
            f"{header.preprint_doi}: the baseline index withheld nothing while the twin "
            "is a corpus member — the arms would not be comparable"
        )
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

    def _result(
        name: str, texts: list[str], tokens: int, wall: float, detail: str,
        dropped: int, arm_excluded: list[str],
    ) -> SystemResult:
        hit = [e.error_id for e in planted.errors if detect(e, texts)]
        return SystemResult(
            system=name,
            detected=hit,
            missed=[e.error_id for e in planted.errors if e.error_id not in hit],
            total_tokens=tokens,
            wall_s=round(wall, 1),
            detail=detail,
            excluded_docs=arm_excluded,
            dropped_chunks=dropped,
        )

    return PlantedEvalReport(
        manuscript=manuscript_path.name,
        twin_in_corpus=twin_present,
        n_errors=len(planted.errors),
        error_kinds={e.error_id: e.kind for e in planted.errors},
        results=[
            # Each arm reports ITS OWN exclusion state (D12). Passing the baseline
            # index's numbers for both made the rows incapable of disagreeing, so a
            # panel that stopped excluding would still have been reported as excluding.
            _result(
                "panel", panel_texts, review.total_tokens, panel_wall,
                f"{len(review.reviewer_outputs)} reviewers · {len(review.verdicts)} claims "
                f"verified · {len(review.deterministic_findings)} deterministic findings",
                review.dropped_chunks, review.excluded_docs,
            ),
            _result(
                "single-agent-equal-compute", baseline.finding_texts, baseline.total_tokens,
                baseline_wall, f"{baseline.samples} self-consistency samples",
                # Both arms report what their OWN index actually withheld, not what
                # was requested — otherwise the two rows describe different things
                # and cannot be compared, which is the point of recording them.
                index.dropped_chunk_count, sorted(index.excluded_docs),
            ),
        ],
        note=_budget_note(
            review.total_tokens, baseline.total_tokens, twin_present
        ),
    )


def render_table(report: PlantedEvalReport) -> str:
    exclusion = (
        f"twin withheld from both arms ({report.results[0].dropped_chunks} chunks)"
        if report.twin_in_corpus
        else "subject held out of the corpus entirely — nothing to withhold"
    )
    lines = [
        f"Planted-error detection · {report.manuscript} · {report.n_errors} errors",
        f"  exclusion: {exclusion}",
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
