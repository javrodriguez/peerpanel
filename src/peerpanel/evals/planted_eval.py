"""The headline measurement: planted-error detection, panel vs single-agent baselines.

Three arms, one protocol. The panel is a two-family mixture by design
(`PanelProviders.local_default`); each single-agent arm is ONE named local model run
alone, so "pass rates for both local models" has a measured rate PER MODEL rather than
one number over a mixture. Every arm reads the same perturbed excerpt and searches the
same index with the same exclusions; every arm is asked for a `quote` and scored on its
own prose through ONE function (`scored_text`); every arm commits the raw strings it
was scored on.

Two counts ship per arm, and which one is the headline is not negotiable:

- `detected` — the assertion rule (`planted.asserts`, `DETECTION_RULE`): some sentence
  of the finding's OWN prose names the planted token AND asserts a defect. This is the
  published number, even when it is 0.
- `named_token` — the older substring rule (`planted.detect`): a finding merely names
  the token somewhere. Published beside the headline, labelled, as an UPPER bound,
  because round 3 found that every catch credited under this rule alone was a verbatim
  quotation of the perturbed manuscript sentence.

`detected` is a subset of `named_token` by construction (asserting requires naming),
and the conformance test recomputes both from the committed `finding_texts`.

What is NOT equal between the arms is recorded rather than smoothed over: the panel
retrieves more (three queries per reviewer plus per-claim verifier retrieval, against
each baseline's single title lookup) and a baseline stops at MAX_SAMPLES, which on
every committed run came before it matched the panel's tokens — so the token column is
derived and the note says, per baseline arm, which had less. Whatever the delta is,
it ships.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel

from peerpanel.agents.reviewer_base import EXCERPT_WORDS, excerpt
from peerpanel.agents.schemas import CallStats
from peerpanel.corpus.models import CorpusManifest
from peerpanel.embeddings import store
from peerpanel.evals.baseline import run_baseline
from peerpanel.evals.planted import (
    DETECTION_RULE,
    PlantedError,
    asserts,
    detect,
    plant_errors,
    scored_text,
)
from peerpanel.graph.pipeline import chunks_for, embedding_fixture_path
from peerpanel.manuscripts.store import read_manuscript
from peerpanel.manuscripts.twins import load_twins
from peerpanel.orchestration.panel import PanelProviders, run_panel
from peerpanel.providers.base import ChatProvider, EmbedProvider
from peerpanel.retrieval import BM25Retriever, Index, VectorRetriever, rrf

PANEL_SYSTEM = "panel"
BASELINE_SYSTEM = "single-agent-cot-sc"

# The named local models the evaluation must publish a pass rate for. Each gets its
# OWN single-agent arm in the same record: a rate measured over a two-family mixture
# is not a rate for either model, and the panel row already names its mixture.
BASELINE_MODELS = ("llama3.1:8b", "qwen2:7b")


def baseline_system(model: str) -> str:
    """The arm name for one single-agent model: `single-agent-cot-sc:llama3.1:8b`."""
    return f"{BASELINE_SYSTEM}:{model}"


def baseline_model_of(provider: ChatProvider) -> str:
    """The model id inside a wire-qualified provider name.

    Providers name themselves `<wire>:<model>` (`ollama-native:llama3.1:8b`), and an
    arm is named for the MODEL: the wire is a run condition the record carries in
    `models`, not part of the arm's identity — naming the arm after the wire would
    make the two published rates differ in a string that is equal for both. A name
    carrying no wire prefix is its own model id, so a stub is still nameable.
    """
    _wire, _, model = provider.name.partition(":")
    return model or provider.name


class SystemResult(BaseModel):
    system: str
    # THE HEADLINE. Errors credited by the assertion rule (planted.asserts): a
    # sentence of this arm's own prose named the token and asserted a defect.
    detected: list[str]  # error_ids
    # The labelled UPPER bound: errors whose token some finding merely named
    # (planted.detect). Required, never defaulted — a record carrying only the
    # headline cannot be checked for the gap between assertion and restatement,
    # and that gap is where round 3 found the entire published number living.
    named_token: list[str]  # error_ids; a superset of `detected` by construction
    missed: list[str]  # the complement of `detected`: what the headline did not credit
    total_tokens: int
    wall_s: float
    detail: str
    excluded_docs: list[str]  # what this arm's index withheld
    dropped_chunks: int  # and how many chunks that actually removed
    finding_texts: list[str]  # the exact strings this arm was scored on
    model_calls: CallStats  # every prompt read whole iff smallest_headroom_tokens > 0
    # Calls that produced nothing because their JSON would not parse: for the panel, a
    # reviewer that came back empty; for the baseline, a self-consistency sample that
    # did. Recorded because it silently lowers the arm's detection count — a panel whose
    # novelty reviewer failed is a one-reviewer panel, and the headline number of this
    # repository would otherwise report that as a two-reviewer result.
    unparsed_calls: int
    # Which model served which role, from the record alone: a reader could previously
    # learn the reviewers' models from the panel record and nothing about the verifier,
    # the converger or the baseline except from prose.
    models: dict[str, str]


class PlantedEvalReport(BaseModel):
    manuscript: str
    twin_in_corpus: bool  # was there anything to exclude at all?
    n_errors: int
    error_kinds: dict[str, str]  # error_id -> kind
    errors: list[PlantedError]  # what was planted, with the token each catch must name
    skipped_kinds: dict[str, str]  # kinds that could NOT be planted, and why
    # The rule `detected` was computed under. Required: a number whose rule is inferred
    # from the commit date is the shape that let a quotation count as a detection for
    # three published rounds.
    detection_rule: str
    results: list[SystemResult]
    note: str


SCORING_NOTE = (
    "Scored over the channels an arm AUTHORS — reviewer findings, REFUTES verdicts and "
    "deterministic-lens findings for the panel; model-authored findings for the "
    "baseline — and, within a finding, over its own prose only. The manuscript text a "
    "finding quotes is committed beside it but never scored: the planted token sits in "
    "the manuscript, so scoring the quote credited an arm for reproducing the perturbed "
    "sentence, which is what every credited detection in the first committed records "
    "turned out to be. THE HEADLINE IS 'detected' under rule "
    f"'{DETECTION_RULE}': some sentence of the finding's own prose both names the "
    "planted token and asserts that something is wrong, with a negation in the five "
    "words before the assertion cue disqualifying it. It is the published number even "
    "when it is 0. Beside it, 'named the token' is the older substring rule, published "
    "as a LABELLED UPPER BOUND and never instead of the headline: every credited "
    "assertion also named the token, so the gap between the two counts is restatement. "
    "Both rules err in stated directions — the assertion rule does not credit a finding "
    "that asserts the defect without naming the token, and does credit a cue used about "
    "something else in the same sentence as the token; the substring rule credits a "
    "finding that names the token while asserting nothing. Every string either rule ran "
    "over is committed in this record under finding_texts; read them."
)


def _budget_note(panel_tokens: int, baseline_tokens: dict[str, int], twin_in_corpus: bool) -> str:
    """The note is DERIVED, never asserted: an earlier hardcoded version claimed a
    matched budget the run's own numbers contradicted. One clause PER BASELINE ARM,
    because with more than one single-agent model a single share sentence describes
    whichever arm the reader guessed."""
    clauses: list[str] = []
    for system, tokens in baseline_tokens.items():
        share = tokens / panel_tokens if panel_tokens else 0.0
        clause = f"{system} {tokens} ({share:.0%} of the panel's)"
        if share < 0.9:
            clause += (
                " — this arm hit its sampling ceiling before matching the panel, so it "
                "is NOT an equal-compute comparison: the baseline had less"
            )
        clauses.append(clause)
    budget = "Token spend: panel " + str(panel_tokens) + "; " + "; ".join(clauses) + "."
    mixture = (
        "The panel row is a two-family mixture — its record names the model that served "
        "each role — while each single-agent row is ONE named model run alone under the "
        "same protocol, so its rate is that model's rate and nobody else's."
    )
    exclusion = (
        "Every arm searched the same index with the same exclusions."
        if twin_in_corpus
        else "The subject is held out of this corpus entirely, so no arm had anything to "
        "withhold and no answer key existed for any of them."
    )
    return f"{SCORING_NOTE} {mixture} {exclusion} {budget}"


def run_planted_eval(
    root: Path,
    manuscript_path: Path,
    providers: PanelProviders,
    baseline_providers: list[ChatProvider],
    embedder: EmbedProvider,
    corpus: str = "demo",
) -> PlantedEvalReport:
    # Named before the panel runs, so a duplicate or missing arm costs nothing: a
    # record whose two baseline rows share a name cannot report either model's rate,
    # and a record with no baseline row is not a comparison at all.
    arm_names = [baseline_system(baseline_model_of(p)) for p in baseline_providers]
    if not arm_names:
        raise ValueError(
            "run_planted_eval needs at least one baseline arm — a panel-only record "
            "is not a comparison"
        )
    if len(set(arm_names)) != len(arm_names):
        raise ValueError(
            f"two baseline arms would carry the same name: {arm_names} — the record "
            "could not tell the models apart"
        )

    header, body = read_manuscript(manuscript_path)
    excluded = load_twins(root / "manuscripts" / "twins.json").excluded_pmcids(header.preprint_doi)
    # Plant only where the reviewers actually look: every arm reads excerpt(body).
    planted = plant_errors(body, manuscript_path.name, window_words=EXCERPT_WORDS)
    visible = excerpt(planted.text)
    unreachable = [e.error_id for e in planted.errors if e.detection_token not in visible]
    if unreachable:
        raise RuntimeError(
            f"planted errors outside the reviewed window: {unreachable} — no arm "
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
        [scored_text(f.text, f.quote) for o in review.reviewer_outputs for f in o.findings]
        + [v.claim_text for v in review.verdicts if v.verdict == "REFUTES"]
        + [f"{d.check} {d.detail}" for d in review.deterministic_findings]
    )

    chunks = chunks_for(root, corpus)
    _chunk_ids, vectors = store.load(embedding_fixture_path(root, corpus))
    # A baseline MUST withhold exactly what the panel withheld. Handing one arm the
    # manuscript's published twin — which states every planted fact correctly — would
    # not be an equal-compute comparison, it would be an answer key.
    index = Index.build(chunks, vectors.astype("float32"), exclude_docs=excluded)
    manifest_rel = "ci.manifest.json" if corpus == "ci" else "demo.manifest.json"
    twin_present = bool(excluded & CorpusManifest.load(root / "corpus" / manifest_rel).pmcids())
    if twin_present and not index.dropped_chunk_count:
        raise RuntimeError(
            f"{header.preprint_doi}: the baseline index withheld nothing while the twin "
            "is a corpus member — the arms would not be comparable"
        )
    chunk_texts = {c.chunk_id: c.text for c in chunks}
    # Each baseline gets the strong non-graph rung (RRF hybrid) — the graph is the
    # panel's advantage to demonstrate, not a handicap to impose on the baseline.
    bm25 = BM25Retriever(index)
    vector = VectorRetriever(index, embedder)

    def retrieve(query: str) -> list[tuple[str, str]]:
        fused = rrf([bm25.search(query, k=8), vector.search(query, k=8)], k=5)
        return [(h.chunk_id, chunk_texts.get(h.chunk_id, "")) for h in fused]

    def _result(
        name: str,
        texts: list[str],
        tokens: int,
        wall: float,
        detail: str,
        dropped: int,
        arm_excluded: list[str],
        calls: CallStats,
        unparsed: int,
        models: dict[str, str],
    ) -> SystemResult:
        # BOTH rules, over the SAME strings, in the one place a score is computed:
        # the headline and its upper bound can never be derived from different text.
        hit = [e.error_id for e in planted.errors if asserts(e, texts)]
        named = [e.error_id for e in planted.errors if detect(e, texts)]
        return SystemResult(
            system=name,
            detected=hit,
            named_token=named,
            missed=[e.error_id for e in planted.errors if e.error_id not in hit],
            total_tokens=tokens,
            wall_s=round(wall, 1),
            detail=detail,
            excluded_docs=arm_excluded,
            dropped_chunks=dropped,
            finding_texts=texts,
            model_calls=calls,
            unparsed_calls=unparsed,
            models=models,
        )

    results = [
        # Each arm reports ITS OWN exclusion state (D12). Passing the baseline
        # index's numbers for both made the rows incapable of disagreeing, so a
        # panel that stopped excluding would still have been reported as excluding.
        _result(
            PANEL_SYSTEM,
            panel_texts,
            review.total_tokens,
            panel_wall,
            f"{sum(1 for o in review.reviewer_outputs if not o.truncated)} of "
            f"{len(review.reviewer_outputs)} reviewers returned findings · "
            f"{len(review.verdicts)} claims verified · "
            f"{len(review.deterministic_findings)} deterministic findings",
            review.dropped_chunks,
            review.excluded_docs,
            review.model_calls,
            sum(1 for o in review.reviewer_outputs if o.truncated),
            review.models,
        )
    ]
    baseline_tokens: dict[str, int] = {}
    for provider, arm_name in zip(baseline_providers, arm_names, strict=True):
        t1 = time.monotonic()
        baseline = run_baseline(
            manuscript_text=planted.text,
            retrieve=retrieve,
            title=header.title,
            provider=provider,
            # Every baseline arm is given the SAME budget — the panel's spend — so
            # the arms differ in model and architecture, never in what they were
            # allowed to spend.
            target_tokens=review.total_tokens,
        )
        baseline_wall = time.monotonic() - t1
        results.append(
            _result(
                arm_name,
                baseline.finding_texts,
                baseline.total_tokens,
                baseline_wall,
                f"{baseline.samples - baseline.unparsed_samples} of {baseline.samples} "
                "self-consistency samples parsed",
                # Both arms report what their OWN index actually withheld, not what
                # was requested — otherwise the two rows describe different things
                # and cannot be compared, which is the point of recording them.
                index.dropped_chunk_count,
                sorted(index.excluded_docs),
                baseline.model_calls,
                baseline.unparsed_samples,
                {"single-agent": provider.name},
            )
        )
        baseline_tokens[arm_name] = baseline.total_tokens

    return PlantedEvalReport(
        manuscript=manuscript_path.name,
        twin_in_corpus=twin_present,
        n_errors=len(planted.errors),
        error_kinds={e.error_id: e.kind for e in planted.errors},
        errors=planted.errors,
        skipped_kinds=planted.skipped_kinds,
        detection_rule=DETECTION_RULE,
        results=results,
        note=_budget_note(review.total_tokens, baseline_tokens, twin_present),
    )


def render_table(report: PlantedEvalReport) -> str:
    exclusion = (
        f"twin withheld from every arm ({report.results[0].dropped_chunks} chunks)"
        if report.twin_in_corpus
        else "subject held out of the corpus entirely — nothing to withhold"
    )
    header = (
        f"  {'system':32s} {'asserted':>11s} {'named token':>13s} "
        f"{'tokens':>9s} {'wall_s':>8s}  detail"
    )
    lines = [
        f"Planted-error detection · {report.manuscript} · {report.n_errors} errors",
        f"  exclusion: {exclusion}",
        f"  rule: {report.detection_rule} — 'asserted' IS the headline, even at 0; "
        "'named token' is the labelled upper bound",
        "",
        header,
    ]
    for result in report.results:
        asserted_cell = f"{len(result.detected)}/{report.n_errors}"
        named_cell = f"{len(result.named_token)}/{report.n_errors}"
        lines.append(
            f"  {result.system:32s} {asserted_cell:>11s} {named_cell:>13s} "
            f"{result.total_tokens:>9d} {result.wall_s:>8.1f}  {result.detail}"
        )
    lines.append("")
    for result in report.results:
        asserted = [report.error_kinds[e] for e in result.detected]
        named = [report.error_kinds[e] for e in result.named_token]
        lines.append(
            f"  {result.system} asserted: {', '.join(asserted) or '(none)'} · "
            f"named the token: {', '.join(named) or '(none)'}"
        )
    if report.skipped_kinds:
        lines.append("")
        for kind, why in sorted(report.skipped_kinds.items()):
            lines.append(f"  not planted: {kind} ({why})")
    lines.append("")
    lines.append(f"  {report.note}")
    return "\n".join(lines)
