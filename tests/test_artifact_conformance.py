"""Committed artifacts must be readable by the current code, self-consistent, and
recomputable from their own contents.

`test_published_numbers.py` binds prose to artifacts. Nothing bound artifacts to
the CODE, so a record produced before a schema change kept sitting in `results/`
describing a state the current implementation cannot emit — a published record
that says exclusion withheld a document while also saying no such document was
in the corpus. Both halves were once true, at different commits.

These tests find every committed record by glob (a new run is covered the moment
it lands), load each through the live models, and re-derive every aggregate the
record carries from the rows it carries. A record whose summary disagrees with
its own rows was not produced by this code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import ValidationError

from peerpanel.agents.claims_verifier import (
    MIN_SWAP_N,
    SOURCE_MANUSCRIPT,
    claim_source,
    swap_consistency_by_source,
    swap_consistency_rate,
)
from peerpanel.agents.reviewer_base import EXCERPT_WORDS, MAX_FINDINGS
from peerpanel.agents.schemas import (
    RUBRIC_DIMENSIONS,
    VERDICT_NEI,
    VERDICT_REFUTES,
    VERDICTS,
    CallStats,
    PanelReview,
)
from peerpanel.embeddings.query_cache import CachedQueryEmbedder
from peerpanel.evals.ablation import RUN_VARYING_FIELDS, run_ablation
from peerpanel.evals.planted import DETECTION_RULE, asserts, detect, plant_errors
from peerpanel.evals.planted_eval import (
    BASELINE_MODELS,
    PANEL_SYSTEM,
    PlantedEvalReport,
    baseline_system,
    scored_residue,
)
from peerpanel.evals.records import assert_run_conditions
from peerpanel.graph.extract import EXTRACTION_PROVIDER_NAME
from peerpanel.manuscripts.store import read_manuscript
from peerpanel.orchestration.panel import detect_conflicts

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PANEL_RECORDS = sorted(p.name for p in RESULTS.glob("panel-review-*.json"))
PLANTED_RECORDS = sorted(p.name for p in RESULTS.glob("planted-eval-*.json"))


def _panel(name: str) -> PanelReview:
    return PanelReview.model_validate_json((RESULTS / name).read_text())


def _planted(name: str) -> PlantedEvalReport:
    """The record, or a failure that says what to do about it.

    A record that predates a schema change is not a bug in the reader — it is a record
    that has to be REGENERATED rather than hand-edited (D18), and the raw pydantic
    traceback says which fields are missing but not that. Both are kept: the message
    names the road, the original error is quoted underneath it.
    """
    try:
        return PlantedEvalReport.model_validate_json((RESULTS / name).read_text())
    except ValidationError as exc:
        raise AssertionError(
            f"{name} does not load under the current schema, so it was produced by "
            "older code: regenerate it (`make eval`, `make eval SUBJECT=met17-auxotroph`) "
            f"rather than editing the record.\n{exc}"
        ) from exc


def _perturbed_for(report: PlantedEvalReport) -> str:
    """The perturbed manuscript this record's arms read, re-derived from the committed
    manuscript by the committed generator.

    Planting is seeded (`planted.SEED`), so this is reproducible rather than stored —
    and the errors it produces must be the ones the record commits, or the record was
    made from a different manuscript than the one in this tree and nothing downstream
    of it means what it says.
    """
    _header, body = read_manuscript(ROOT / "manuscripts" / report.manuscript)
    planted = plant_errors(body, report.manuscript, window_words=EXCERPT_WORDS)
    assert planted.errors == report.errors, (
        f"{report.manuscript}: re-planting this manuscript does not reproduce the "
        "errors the record carries — the record and the manuscript in this tree "
        "describe different runs"
    )
    return planted.text


BUILD_RECORDS = sorted(p.name for p in RESULTS.glob("build-stats-*.json"))


def test_there_are_records_to_hold_to_account() -> None:
    """Every capability the prose calls measured needs at least one committed run.

    Build records are named here too: they are globbed at import, and a parametrize over
    an empty list SKIPS rather than fails — so without this line a repository that had
    lost every index build would go green with seven quiet skips.
    """
    assert PANEL_RECORDS, "no committed panel run under results/"
    assert PLANTED_RECORDS, "no committed planted-error evaluation under results/"
    assert BUILD_RECORDS, "no committed index build under results/"


@pytest.mark.parametrize("name", PANEL_RECORDS)
class TestPanelRecords:
    def test_loads_under_the_current_schema(self, name: str) -> None:
        _panel(name)

    def test_reviewer_outputs_are_within_the_code_s_own_bounds(self, name: str) -> None:
        for output in _panel(name).reviewer_outputs:
            assert len(output.findings) <= MAX_FINDINGS, output.reviewer
            if output.truncated:
                # A reviewer whose two attempts both failed to parse emits exactly this
                # shape (reviewer_base): no scores, no confidence, no findings. Holding
                # it to the scored bounds would refuse a record this code can produce —
                # and the honest reading of a failed reviewer is that it scored nothing.
                assert output.scores == {} and output.confidence == 0, output.reviewer
                assert output.findings == [], output.reviewer
                continue
            assert 1 <= output.confidence <= 5, output.reviewer
            assert set(output.scores) == set(RUBRIC_DIMENSIONS), output.reviewer
            assert all(1 <= s <= 5 for s in output.scores.values()), output.reviewer
            for finding in output.findings:
                assert finding.dimension in RUBRIC_DIMENSIONS, output.reviewer
                assert finding.severity in {"major", "minor"}, output.reviewer
                assert finding.text.strip(), output.reviewer

    def test_an_unswapped_claim_is_an_abstention_with_no_evidence(self, name: str) -> None:
        """The code's unswapped branch emits exactly one shape: NEI, no span,
        trivially consistent. Anything else under swapped=False was not its output."""
        for v in _panel(name).verdicts:
            assert v.verdict in VERDICTS, v.claim_id
            if not v.swapped:
                assert v.verdict == VERDICT_NEI and not v.evidence and v.swap_consistent, (
                    f"{v.claim_id}: unswapped but {v.verdict} / {len(v.evidence)} spans"
                )

    def test_swap_aggregates_recompute_from_the_verdict_rows(self, name: str) -> None:
        review = _panel(name)
        rate, n = swap_consistency_rate(review.verdicts)
        assert (review.swap_consistency_rate, review.n_swap_checked) == (rate, n)
        assert n == sum(1 for v in review.verdicts if v.swapped)
        by_source = {k: (s.rate, s.n) for k, s in review.swap_by_source.items()}
        assert by_source == swap_consistency_by_source(review.verdicts)
        for source, (source_rate, source_n) in by_source.items():
            assert (source_rate is None) == (source_n < MIN_SWAP_N), source

    def test_claim_sources_are_the_manuscript_or_a_reviewer_that_ran(self, name: str) -> None:
        review = _panel(name)
        reviewers = {o.reviewer for o in review.reviewer_outputs}
        for v in review.verdicts:
            source = claim_source(v.claim_id)
            assert source == SOURCE_MANUSCRIPT or source in reviewers, v.claim_id
            assert ":panel000" not in v.claim_id, "the retired verbatim-opinion claim id"
        assert set(review.swap_by_source) <= {SOURCE_MANUSCRIPT, *reviewers}

    def test_conflicts_recompute_from_the_rows(self, name: str) -> None:
        """detect_conflicts is a pure function of the reviewer outputs and verdicts,
        so the committed conflicts must be exactly what it returns for them."""
        review = _panel(name)
        recomputed = detect_conflicts(review.reviewer_outputs, review.verdicts)
        assert [c.model_dump() for c in recomputed] == [c.model_dump() for c in review.conflicts]
        evidence_conflicts = sum(1 for c in review.conflicts if c.type == "evidence")
        grounded_refutes = sum(
            1
            for v in review.verdicts
            if v.verdict == VERDICT_REFUTES
            and v.evidence
            and claim_source(v.claim_id) != SOURCE_MANUSCRIPT
        )
        assert evidence_conflicts == grounded_refutes

    def test_exclusion_state_is_consistent(self, name: str) -> None:
        review = _panel(name)
        assert (review.dropped_chunks > 0) == bool(review.excluded_docs)


@pytest.mark.parametrize("name", PLANTED_RECORDS)
class TestPlantedEvalRecords:
    def test_loads_under_the_current_schema(self, name: str) -> None:
        _planted(name)

    def test_the_arms_are_the_systems_the_code_names(self, name: str) -> None:
        """One panel row and one single-agent row PER NAMED MODEL (DECISIONS D-2).

        The panel is a two-family mixture by design, so its row names a mixture and
        cannot carry a pass rate for either model on its own; the single-agent arm runs
        once per model, in the same record, so every model this repository names has a
        rate of its own. A record with fewer arms than the code names is a record whose
        table cannot say what its heading says.
        """
        expected = {PANEL_SYSTEM, *(baseline_system(m) for m in BASELINE_MODELS)}
        assert {arm.system for arm in _planted(name).results} == expected

    def test_detection_recomputes_from_the_scored_texts(self, name: str) -> None:
        """Every arm commits the exact strings it was scored on. Re-running BOTH rules
        over them must reproduce both counts to the id.

        Two counts, two rules, one direction between them. `detected` is the published
        headline and credits a finding whose own prose asserts the defect its planted
        record names (`asserts`, rule `DETECTION_RULE`); `named_token` is the labelled
        upper bound and credits a finding that merely names the token (`detect`). An
        assertion names the token by construction, so the headline can never exceed its
        own upper bound — a record where it does was produced by neither rule.

        Both rules are re-run over `scored_texts`, the residue, because that is what the
        harness scores under `assertion-v2`: `finding_texts` still holds everything the
        arm wrote, and recomputing from it would credit the quotation the residue
        removes — the round-4 defect this record shape exists to close.
        """
        report = _planted(name)
        assert report.detection_rule == DETECTION_RULE, (
            f"{name}: scored under {report.detection_rule!r}, which is not the rule this "
            f"code credits ({DETECTION_RULE!r})"
        )
        ids = [e.error_id for e in report.errors]
        assert report.n_errors == len(ids)
        assert report.error_kinds == {e.error_id: e.kind for e in report.errors}
        for arm in report.results:
            detected = [e.error_id for e in report.errors if asserts(e, arm.scored_texts)]
            missed = [e.error_id for e in report.errors if not asserts(e, arm.scored_texts)]
            named = [e.error_id for e in report.errors if detect(e, arm.scored_texts)]
            assert (arm.detected, arm.missed) == (detected, missed), arm.system
            assert arm.named_token == named, arm.system
            assert set(arm.detected) | set(arm.missed) == set(ids), arm.system
            assert not set(arm.detected) & set(arm.missed), arm.system
            assert set(arm.detected) <= set(arm.named_token), (
                f"{arm.system}: {sorted(set(arm.detected) - set(arm.named_token))} credited as "
                "asserting a defect whose token the upper bound says no finding named"
            )

    def test_the_scored_texts_are_the_committed_finding_texts_stripped(
        self, name: str
    ) -> None:
        """The residue must be DERIVABLE from what the record already commits.

        A record that simply asserted "these are the strings we scored" would be back
        where round 4 found it: a description no reader can check. So every scored
        string is re-derived here from the arm's own `finding_texts` and the perturbed
        manuscript this tree's generator plants, through the same `scored_residue` the
        harness used — one rule, two callers.
        """
        report = _planted(name)
        perturbed = _perturbed_for(report)
        for arm in report.results:
            expected, dropped = scored_residue(arm.finding_texts, perturbed)
            assert arm.scored_texts == expected, (
                f"{arm.system}: the committed residue is not what stripping this "
                "record's own finding_texts produces"
            )
            assert arm.quoted_sentences_dropped == dropped, arm.system
            assert len(arm.scored_texts) <= len(arm.finding_texts), (
                f"{arm.system}: stripping produced MORE strings than the arm wrote"
            )
            for error in report.errors:
                if detect(error, arm.scored_texts):
                    assert detect(error, arm.finding_texts), (
                        f"{arm.system}: {error.error_id} is credited on the residue and "
                        "not on the full text — stripping can only ever lower a count"
                    )

    def test_a_kind_is_planted_or_skipped_never_both(self, name: str) -> None:
        report = _planted(name)
        planted_kinds = set(report.error_kinds.values())
        assert not planted_kinds & set(report.skipped_kinds), (
            f"kinds both planted and reported unplantable: "
            f"{planted_kinds & set(report.skipped_kinds)}"
        )

    def test_exclusion_state_cannot_contradict_itself(self, name: str) -> None:
        """The defect this exists for: a record naming an excluded document while
        also recording that no such document was in the corpus."""
        report = _planted(name)
        for arm in report.results:
            if not report.twin_in_corpus:
                assert not arm.excluded_docs, (
                    f"{arm.system} names {arm.excluded_docs} as excluded, but the record "
                    "says the twin is not a corpus member — the current code cannot emit this"
                )
            assert (arm.dropped_chunks > 0) == bool(arm.excluded_docs), (
                f"{arm.system}: excluded_docs and dropped_chunks disagree"
            )

    def test_every_arm_reports_the_same_exclusion(self, name: str) -> None:
        states = {(tuple(a.excluded_docs), a.dropped_chunks) for a in _planted(name).results}
        assert len(states) == 1, f"the arms searched different indexes: {states}"


class TestAblationRecordIsThisCodesOutput:
    def test_ci_ladder_regenerates_byte_for_byte_except_latency(self) -> None:
        """The committed CI ablation is a deterministic function of committed bytes
        (corpus, extractions, query vectors). Regenerate it here and compare everything
        but the fields the schema itself DECLARES run-varying: any other difference
        means the record was not produced by this code.

        The declared list is imported rather than repeated, so a field that becomes
        run-varying is exempted in one place and a field that stops being run-varying
        cannot stay quietly exempt here. `run_utc` is then checked to have MOVED —
        checklist line 3's "the regenerated record's run timestamp moved" — because a
        stripped field nobody looks at is a record that need never have been re-run.
        """
        committed = json.loads((RESULTS / "ablation-ci.json").read_text())
        fresh = run_ablation(ROOT, CachedQueryEmbedder(ROOT), corpus="ci").model_dump()

        def strip(report: dict[str, object]) -> dict[str, object]:
            rows = report["results"]
            assert isinstance(rows, list)
            return {
                **{k: v for k, v in report.items() if k not in RUN_VARYING_FIELDS},
                "results": [
                    {k: v for k, v in row.items() if k not in RUN_VARYING_FIELDS} for row in rows
                ],
            }

        assert strip(fresh) == strip(committed)
        assert "run_utc" in committed, (
            "the committed ladder carries no run timestamp, so nothing can show it moved; "
            "regenerate it (`make ablation-publish`)"
        )
        assert fresh["run_utc"] != committed["run_utc"], (
            f"a fresh ladder came back stamped {fresh['run_utc']}, the committed record's own "
            "timestamp — the regeneration this test claims to have run did not happen"
        )


_WORD_NUMBERS = {
    word: n
    for n, word in enumerate(
        [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
        ]
    )
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _as_number(token: str) -> int | None:
    """The prose writes small counts as words ("three hallucinated findings in sixteen")."""
    token = token.strip().lower()
    return int(token) if token.isdigit() else _WORD_NUMBERS.get(token)


def _read_whole(
    stats: CallStats,
    where: str,
    *,
    calls_required: bool = True,
    record: dict[str, object] | None = None,
) -> None:
    """The D19 proof, asked through the ONE rule: `evals.records.assert_run_conditions`.

    The body used to live here, which meant the requirement-8 sweep over every
    `results/**/*.json` could only re-implement it — and two implementations of one rule
    drift until a record passes one and fails the other. The rule, its rationale and its
    messages are in `peerpanel/evals/records.py`; this stays as the local spelling
    (`calls_required=False` is the replay branch) so the tests below read as they did.
    """
    assert_run_conditions(stats, where, replay_allowed=not calls_required, record=record)


class TestEveryPromptWasReadWhole:
    """Before D19 every long prompt in this project was silently cut to a 4,096-token
    window's tail. Each record now carries its calls, its largest prompt, the smallest
    window it ran in, and the smallest margin between a prompt and the window THAT prompt
    ran in. The margin is what is checked: the largest prompt and the smallest window
    usually belong to different calls, so `build-stats-ci.json` legitimately shows a
    5,238-token prompt beside a 4,096-token window and was still read whole on every
    call."""

    @pytest.mark.parametrize("name", PANEL_RECORDS)
    def test_panel_records(self, name: str) -> None:
        review = _panel(name)
        _read_whole(review.model_calls, name)
        # Two reviewers, the verifier's two orderings per claim, the converger: never one call.
        assert review.model_calls.calls >= 2 + len(review.reviewer_outputs), name

    def test_a_panel_that_produced_nothing_is_not_a_panel_result(self) -> None:
        """A reviewer whose JSON will not parse contributes no findings, which lowers the
        detection count of the arm this repository's headline compares. One is a caveat
        the record now carries; all of them is not a panel run at all."""
        for name in PANEL_RECORDS:
            review = _panel(name)
            assert any(not o.truncated for o in review.reviewer_outputs), (
                f"{name}: every reviewer failed to parse — this record measures nothing"
            )

    def test_an_arm_that_lost_calls_says_so_in_the_prose(self) -> None:
        """If an arm silently lost a call, the number it produced is not the number the
        design intends, and a reader comparing the two arms has to be told."""
        results_text = (RESULTS / "RESULTS.md").read_text()
        for name in PLANTED_RECORDS:
            for arm in _planted(name).results:
                if arm.unparsed_calls:
                    assert "unparsed" in results_text or "did not parse" in results_text, (
                        f"{name}: the {arm.system} arm lost {arm.unparsed_calls} call(s) to "
                        "unparseable output; RESULTS.md must say so beside the count"
                    )

    @pytest.mark.parametrize("name", PLANTED_RECORDS)
    def test_every_arm_of_the_planted_evaluation(self, name: str) -> None:
        for arm in _planted(name).results:
            _read_whole(arm.model_calls, f"{name}:{arm.system}")

    @pytest.mark.parametrize("name", BUILD_RECORDS)
    def test_index_builds(self, name: str) -> None:
        record = json.loads((RESULTS / name).read_text())
        assert record["provider"] == EXTRACTION_PROVIDER_NAME, (
            f"{name}: extractions filed under {record['provider']!r}; the cache identity "
            f"the code replays is {EXTRACTION_PROVIDER_NAME!r}"
        )
        # A build record may legitimately show no calls: `make graph` on the committed
        # cache replays every chunk and calls nothing. What it may not do is report
        # measurements it did not make, or a chunk it recorded as truncated.
        assert record["chunks"] > 0, f"{name}: a zero-chunk build proves nothing"
        stats = CallStats.model_validate(record["model_calls"])
        # The replay branch needs the record itself: a build that called nothing has to
        # show every chunk was already on disk. That check, and why only this direction
        # of it is sound, moved into the one rule with the rest of it.
        _read_whole(stats, name, calls_required=False, record=record)
        assert record["truncated_chunks"] == 0, f"{name}: a chunk recorded as truncated"


# ======================================================================================
# The published yields, and the populations they are published over.
#
# Round 5, report 3, Finding 1: `test_grounding_yield_is_stated_accurately` computed its
# numerator and then asserted it only inside the `if grounded == 0:` branch, so while the
# yield was non-zero the single live assertion was that the string "40" appeared SOMEWHERE
# in RESULTS.md. An evaluator moved ten published numbers with all 594 tests green,
# including this yield and the panel's verdict distribution. Every numerator below is now
# asserted whatever its value.
#
# Round 5, report 1, Finding 4: the two committed panel runs are byte-identical apart from
# `wall_s`, so "12 of 40" and "0 of 32" count ONE population twice. The arithmetic rule
# accepts a doubled pair only as the exact sum of the runs; the presentation rule after it
# requires a doubled form to say that it is a reproduction counted twice, or to be
# restated per run. Neither rule invents a preferred wording — both roads are named in the
# failure message, and the honest per-run figures are what is bound.
#
# Every check is a module-level function taking the prose as an argument, so a control can
# hand it a mutated document and prove the check goes red (D16: a verification that cannot
# fail is worth nothing). The controls are at the bottom of this file.
# ======================================================================================

_BULLET = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")


def _prose_units(text: str) -> list[str]:
    """Prose paragraphs: table rows dropped, bullets kept apart, emphasis stripped.

    Table cells are bound structurally in `test_published_numbers.py` (row label x column
    header). Folding them in here would join eleven unrelated rows into one "sentence",
    and a claim would then borrow the caveat of a row three lines away — the same reason
    a bullet ends a sentence. Emphasis and backticks go because a claim wrapped in `**`
    would otherwise never end a sentence.
    """
    units: list[str] = []
    current: list[str] = []

    def flush() -> None:
        joined = re.sub(r"\s+", " ", " ".join(current)).strip()
        if joined:
            units.append(joined)
        current.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("|") or stripped.startswith("#"):
            flush()
            continue
        if _BULLET.match(line):
            flush()
        current.append(re.sub(r"[*`_]", "", line))
    flush()
    return units


def _sentences(text: str) -> list[str]:
    """Every prose sentence, in document order."""
    return [sentence for _unit, sentence in _sentences_in_context(text)]


def _sentences_in_context(text: str) -> list[tuple[str, str]]:
    """(the paragraph a sentence sits in, the sentence).

    Two scopes, deliberately. WHICH mechanism a number is about is often established one
    sentence earlier — "Evidence grounding is enforced in code, and it now fires." then
    "on the rebuilt index it is 12 of 40 verdicts" — so classification reads the
    paragraph. Whether a claim is a superseded measurement, and whether a doubled figure
    decomposes itself, are properties of the sentence making the claim: a paragraph must
    never lend its disclaimer to a live figure three sentences later.
    """
    out: list[tuple[str, str]] = []
    for unit in _prose_units(text):
        for sentence in re.split(r"(?<=[.!?])\s+", unit):
            if sentence.strip():
                out.append((unit, sentence))
    return out


# A sentence that says it is reporting a superseded measurement is not bound to today's
# records — that is the whole point of publishing the old number beside the correction.
# The marker has to be IN the sentence making the claim, so a paragraph cannot lend its
# disclaimer to a live figure three sentences later.
_HISTORICAL = re.compile(
    r"an earlier (?:version|run|record|reading)|a previous version|previous version of this|"
    r"earlier version of this|until the rebuild|until this commit|before the rebuild|"
    r"used to (?:say|read|report)|was once",
    re.IGNORECASE,
)

# What makes a doubled population honest: it is named as one population counted twice, or
# it is decomposed per run in the same sentence ("6 in each run", "eight findings each").
_DECOMPOSED = re.compile(r"\beach\b|per run|counted twice|twice over|in both runs of", re.I)
_REPRODUCTION = re.compile(r"reproduc\w+", re.IGNORECASE)
_NOT_A_SAMPLE = re.compile(r"identical|independent sample|second sample|the same run", re.I)


def _names_the_runs_as_one(text: str) -> bool:
    """Does this document say anywhere that the second run reproduces the first?

    Required of a document that publishes a doubled n. `RESULTS.md` says it in as many
    words ("a second run is a reproduction, not an independent sample"); until this round
    `README.md` said it nowhere while publishing 12 of 40 and 0 of 32.
    """
    return any(
        _REPRODUCTION.search(s) and _NOT_A_SAMPLE.search(s) for s in _sentences(text)
    )


def _results_text() -> str:
    return (RESULTS / "RESULTS.md").read_text()


def _readme_text() -> str:
    return (ROOT / "README.md").read_text()


def _quotes_outside_the_manuscript(name: str, review: PanelReview) -> tuple[int, int]:
    """(findings quoting text that is not in the manuscript, findings) for one run."""
    # Two passes on purpose: "-run\d+$" cannot match while ".json" still trails, so a
    # single alternation leaves "met17-auxotroph-run2" and looks for a manuscript by that
    # name.
    stem = re.sub(r"-run\d+$", "", Path(name).stem.replace("panel-review-", "", 1))
    manuscript = _normalise((ROOT / "manuscripts" / f"{stem}.txt").read_text())
    offending = total = 0
    for output in review.reviewer_outputs:
        for finding in output.findings:
            total += 1
            if finding.quote and _normalise(finding.quote) not in manuscript:
                offending += 1
    return offending, total


def _per_run_yields() -> dict[str, list[tuple[int, int]]]:
    """{kind: [(numerator, denominator), one pair per committed run]}.

    Per run, never summed: the runs are a reproduction of one another, and the sum is a
    presentation of that pair rather than an n. The sum is derived where a claim uses it,
    by `_allowed_pairs`, so the two forms can never disagree.
    """
    kinds: dict[str, list[tuple[int, int]]] = {
        "grounding": [],
        "order-change": [],
        "hallucination": [],
        "reviewer citations": [],
    }
    for name in PANEL_RECORDS:
        review = _panel(name)
        kinds["grounding"].append(
            (sum(1 for v in review.verdicts if v.evidence), len(review.verdicts))
        )
        kinds["order-change"].append(
            (sum(1 for v in review.verdicts if not v.swap_consistent), review.n_swap_checked)
        )
        kinds["hallucination"].append(_quotes_outside_the_manuscript(name, review))
        findings = [f for o in review.reviewer_outputs for f in o.findings]
        kinds["reviewer citations"].append(
            (sum(1 for f in findings if f.evidence_chunk_ids), len(findings))
        )
    return kinds


def _allowed_pairs(per_run: list[tuple[int, int]]) -> dict[tuple[int, int], str]:
    """{(numerator, denominator): how it is arrived at} — per run, and the doubled sum."""
    allowed = {pair: "one run" for pair in per_run}
    if len(per_run) > 1:
        doubled = (sum(n for n, _ in per_run), sum(d for _, d in per_run))
        allowed.setdefault(doubled, "every committed run added together")
    return allowed


def _lens_findings_per_run() -> list[int]:
    return [len(_panel(name).deterministic_findings) for name in PANEL_RECORDS]


_KIND_VOCABULARY = {
    "grounding": re.compile(r"evidence span|surviving evidence|grounding|grounded", re.I),
    "order-change": re.compile(r"evidence order|order[- ]swapped|swap|reversed", re.I),
}
_HALLUCINATION = re.compile(r"quot\w+", re.IGNORECASE)
_VERDICT_PAIR = re.compile(r"(\d[\d,]*)\s+of\s+(\d[\d,]*)\s+verdicts?", re.IGNORECASE)
_BARE_PAIR = re.compile(r"(\d[\d,]*)\s+of\s+(\d[\d,]*)")


def _int(token: str) -> int:
    return int(token.replace(",", ""))


def _claims(results_text: str, readme_text: str) -> list[tuple[str, str, str, int, int]]:
    """Every live population claim in the two documents, as
    (kind, where, sentence, numerator, denominator).

    A claim naming a verdict population but no mechanism this test knows is an ERROR
    rather than a skip: an unclassifiable claim is an unbound claim, which is the defect
    this section exists to close.
    """
    found: list[tuple[str, str, str, int, int]] = []
    for where, text in (("results/RESULTS.md", results_text), ("README.md", readme_text)):
        for unit, sentence in _sentences_in_context(text):
            if _HISTORICAL.search(sentence):
                continue
            for match in _VERDICT_PAIR.finditer(sentence):
                kinds = [k for k, vocab in _KIND_VOCABULARY.items() if vocab.search(unit)]
                assert kinds, (
                    f"{where}: {match.group(0)!r} publishes a verdict population, and the "
                    "sentence around it names no mechanism this binding can recompute it "
                    f"from ({sorted(_KIND_VOCABULARY)}). Say which mechanism the number is "
                    f"about, or the number is unbound.\n  {sentence}"
                )
                for kind in kinds:
                    found.append(
                        (kind, where, sentence, _int(match.group(1)), _int(match.group(2)))
                    )
            if _HALLUCINATION.search(sentence) and "manuscript" in sentence.lower():
                for match in _BARE_PAIR.finditer(sentence):
                    found.append(
                        (
                            "hallucination",
                            where,
                            sentence,
                            _int(match.group(1)),
                            _int(match.group(2)),
                        )
                    )
    return found


def _check_published_populations_recompute(results_text: str, readme_text: str) -> int:
    """Every live "N of M" claim about the panel, numerator included, against the records.

    A claim naming several mechanisms is allowed to match any one of them; a claim naming
    one must match that one exactly.
    """
    yields = _per_run_yields()
    checked = 0
    by_claim: dict[tuple[str, str, int, int], list[str]] = {}
    for kind, where, sentence, numerator, denominator in _claims(results_text, readme_text):
        by_claim.setdefault((where, sentence, numerator, denominator), []).append(kind)
    for (where, sentence, numerator, denominator), kinds in by_claim.items():
        allowed: dict[tuple[int, int], str] = {}
        for kind in kinds:
            allowed |= _allowed_pairs(yields[kind])
        assert (numerator, denominator) in allowed, (
            f"{where} publishes {numerator} of {denominator} as the "
            f"{' or '.join(sorted(kinds))} yield; the committed runs give "
            + " and ".join(f"{n} of {d}" for n, d in yields[kinds[0]])
            + f" ({', '.join(f'{n} of {d} — {how}' for (n, d), how in sorted(allowed.items()))}).\n"
            f"  {sentence}"
        )
        checked += 1
    assert checked >= 4, (
        f"only {checked} population claims matched in README.md and results/RESULTS.md. "
        "This binding is bound to nothing, which is round 3's dead-guard class (F6): the "
        "grounding yield, the order-change rate and the hallucination count are each "
        "published in both files, so a shape this regex cannot read is a shape that has "
        "gone unbound."
    )
    return checked


def _check_a_doubled_population_says_so(results_text: str, readme_text: str) -> None:
    """A figure counted over both runs is one population counted twice, and says so.

    Round 5, report 1, Finding 4. The records are byte-identical apart from `wall_s`, so
    "12 of 40" and "0 of 32" state the sample size at twice its true value — in the same
    documents that withhold a rate below an n gate and say "a second run is a
    reproduction, not an independent sample". Two roads out, and this rule takes either:
    restate the figure per run, or keep the doubled form and attach its own caveat.

    Every violation is collected and reported together. A rule that stops at the first
    one makes a prose pass into a queue of single failures, and the writer cannot see the
    shape of what they are fixing.
    """
    yields = _per_run_yields()
    problems: list[str] = []
    for kind, where, sentence, numerator, denominator in _claims(results_text, readme_text):
        per_run = yields[kind]
        if len(per_run) < 2 or (numerator, denominator) in set(per_run):
            continue
        if (numerator, denominator) != (sum(n for n, _ in per_run), sum(d for _, d in per_run)):
            continue  # the arithmetic rule owns a pair that is neither
        text = results_text if where.endswith("RESULTS.md") else readme_text
        detail = (
            f"{where} publishes {numerator} of {denominator} for the {kind} yield. That "
            f"denominator is {len(per_run)} runs added together, and the runs are "
            "byte-identical apart from wall-clock, so it is one population counted twice "
            f"rather than a sample of {denominator}. Per run the records give "
            + " and ".join(f"{n} of {d}" for n, d in per_run)
            + ".\n  "
            + sentence
        )
        if not _DECOMPOSED.search(sentence):
            problems.append(
                detail + "\n  Either restate it per run, or decompose it in the same "
                'sentence ("6 in each run", "eight findings each").'
            )
        elif not _names_the_runs_as_one(text):
            problems.append(
                detail + f"\n  {where} nowhere says the second run reproduces the first, "
                "so a reader sees two samples. Say it in this document (results/RESULTS.md "
                'says "a second run is a reproduction, not an independent sample"), or '
                "publish the per-run figure here."
            )
    assert not problems, (
        f"{len(problems)} doubled population(s) are published as an n:\n\n"
        + "\n\n".join(problems)
    )


_VERDICT_IN_PROSE = re.compile(
    r"(\d[\d,]*)\s+(SUPPORTS|REFUTES|NOT_ENOUGH_INFO|abstentions?)", re.IGNORECASE
)
_POPULATION = re.compile(r"(?:across|of|out of)\s+(\d[\d,]*)", re.IGNORECASE)


def _check_the_verdict_distribution_in_prose(results_text: str, readme_text: str) -> None:
    """"0 SUPPORTS, 0 REFUTES, 20 abstentions per run", against the records.

    The table cell carrying the same distribution is bound by row label and column in
    `test_published_numbers.py`; this is the sentence that repeats it. The evaluator's
    mutation was exactly this — two abstentions turned into two SUPPORTS — and it is the
    most load-bearing honest number on the page: the section "the verifier decides
    nothing" is built on it.

    A doubled population is accepted here without the caveat the "N of M" rule asks for,
    because round 5 filed the doubled-n finding against the six sentences it enumerated
    and this is not one of them. The COUNT is bound either way.
    """
    counts = [
        {
            verdict: sum(1 for v in _panel(name).verdicts if v.verdict == verdict)
            for verdict in VERDICTS
        }
        for name in PANEL_RECORDS
    ]
    assert all(c == counts[0] for c in counts), (
        f"the committed runs publish different verdict distributions {counts}; a single "
        "published distribution cannot describe them"
    )
    per_run = counts[0]
    doubled = {verdict: total * len(counts) for verdict, total in per_run.items()}
    totals = {sum(per_run.values()), sum(doubled.values())}
    seen = 0
    for where, text in (("results/RESULTS.md", results_text), ("README.md", readme_text)):
        for _unit, sentence in _sentences_in_context(text):
            if _HISTORICAL.search(sentence):
                continue
            claims = _VERDICT_IN_PROSE.findall(sentence)
            if len({verdict.upper() for _n, verdict in claims}) < 2:
                continue  # a lone "20 abstentions" is a count, not the distribution
            seen += 1
            for shown, word in claims:
                verdict = VERDICT_NEI if word.lower().startswith("abstention") else word.upper()
                claimed = _int(shown)
                assert claimed in {per_run[verdict], doubled[verdict]}, (
                    f"{where} publishes {shown} {word}; the committed runs record "
                    f"{per_run[verdict]} per run ({doubled[verdict]} over "
                    f"{len(counts)} runs).\n  {sentence}"
                )
            for population in _POPULATION.findall(sentence):
                assert _int(population) in totals, (
                    f"{where} states the verdict distribution over a population of "
                    f"{population}; the committed runs judged "
                    f"{sorted(totals)[0]} claims per run.\n  {sentence}"
                )
    assert seen, (
        "no verdict-distribution sentence matched in README.md or results/RESULTS.md, so "
        f"this guard covers nothing (round 3's F6). The records say {per_run}, and both "
        "files state it in prose beside the table."
    )


def _check_the_lens_yield(results_text: str, readme_text: str) -> None:
    """The deterministic lens's count, wherever it is stated, whatever it is."""
    per_run = _lens_findings_per_run()
    assert len(set(per_run)) == 1, (
        f"the committed runs disagree on the lens count {per_run}; a single published "
        "figure cannot describe them"
    )
    found = per_run[0]
    seen = 0
    for where, text in (("results/RESULTS.md", results_text), ("README.md", readme_text)):
        for sentence in _sentences(text):
            if "lens" not in sentence.lower() or _HISTORICAL.search(sentence):
                continue
            for match in re.finditer(r"(\d[\d,]*|no|zero)\s+findings", sentence, re.IGNORECASE):
                seen += 1
                shown = match.group(1).lower()
                claimed = 0 if shown in {"no", "zero"} else _int(shown)
                assert claimed == found, (
                    f"{where} says {match.group(0)!r} of the deterministic lens; the "
                    f"committed runs record {found} on every run.\n  {sentence}"
                )
    assert seen, (
        "no deterministic-lens count matched in README.md or results/RESULTS.md, so this "
        f"guard covers nothing (round 3's F6). The records say {found} findings on every "
        "committed run, and both files describe the lens as a mechanism that has never "
        "fired — a description that has to carry the number it rests on."
    )
    if found == 0:
        assert "0 findings" in results_text, (
            "results/RESULTS.md must state the lens yield as a zero: a mechanism with no "
            "output on any committed run is a gap, and this page says so in numbers"
        )


def _check_the_reviewer_citation_yield(results_text: str, readme_text: str) -> None:
    """"1 reviewer finding in 16 cites a retrieved chunk" — both numbers, from the records.

    Deliberately shape-agnostic: the two live sentences write the pair in opposite orders
    ("of 16 findings in a run, 1 cites…" and "only 1 reviewer finding in 16 cites…"), so
    this requires both numbers to be PRESENT in the sentence rather than pinning an order.
    It therefore catches a moved value and not a transposed one; the panel table's
    per-reviewer findings row binds the 16 by position.
    """
    per_run = _per_run_yields()["reviewer citations"]
    assert len(set(per_run)) == 1, f"the committed runs disagree on citing findings: {per_run}"
    citing, findings = per_run[0]
    seen = 0
    for where, text in (("results/RESULTS.md", results_text), ("README.md", readme_text)):
        for sentence in _sentences(text):
            if "cites a retrieved chunk" not in sentence.lower() or _HISTORICAL.search(sentence):
                continue
            seen += 1
            numbers = {_int(n) for n in re.findall(r"\d[\d,]*", sentence)}
            assert {citing, findings} <= numbers, (
                f"{where} states the reviewer-citation yield without the numbers the "
                f"records carry: {citing} of {findings} findings in a run cite a "
                f"retrieved chunk, and this sentence carries {sorted(numbers)}.\n  {sentence}"
            )
    assert seen, (
        "no reviewer-citation claim matched in either file; the records say "
        f"{citing} of {findings} and both documents publish it, so this guard has gone "
        "dead rather than the claim having gone away"
    )


class TestClaimedYieldsMatchTheRecords:
    """Two capabilities are described in prose as strengths. Their measured yield
    is bound here so the description can never drift from the artifacts again.

    Every method delegates to a module-level checker taking the two documents as text,
    so `TestTheseYieldBindingsCanGoRed` below can run the same rule over a mutated copy
    and prove the rule fires.
    """

    def test_grounding_yield_is_stated_accurately(self) -> None:
        """The numerator, unconditionally — the assertion Finding 1 found missing."""
        results_text, readme = _results_text(), _readme_text()
        grounded = sum(n for n, _ in _per_run_yields()["grounding"])
        total = sum(d for _, d in _per_run_yields()["grounding"])
        if grounded == 0:
            for text, where in ((results_text, "RESULTS.md"), (readme, "README.md")):
                assert "never" in text.lower() and "zero" in text.lower(), (
                    f"{where} must say the grounding yield is zero while it is "
                    f"({grounded}/{total} spans across committed runs)"
                )
        _check_published_populations_recompute(results_text, readme)

    def test_a_doubled_population_is_published_as_a_reproduction(self) -> None:
        _check_a_doubled_population_says_so(_results_text(), _readme_text())

    def test_the_verdict_distribution_in_prose_matches_the_records(self) -> None:
        _check_the_verdict_distribution_in_prose(_results_text(), _readme_text())

    def test_deterministic_lens_yield(self) -> None:
        _check_the_lens_yield(_results_text(), _readme_text())

    def test_reviewer_citation_yield(self) -> None:
        _check_the_reviewer_citation_yield(_results_text(), _readme_text())

    def test_the_hallucination_count_is_the_records_own_count(self) -> None:
        """The one number the prose reports AGAINST itself, and the only one that was
        published without a binding: reviewer findings quoting text that is not in the
        manuscript under review. Recomputed here from the records, both files required to
        state it, so the count can never be softened by a rewrite or left behind by a
        re-run."""
        per_run = _per_run_yields()["hallucination"]
        offending = sum(n for n, _ in per_run)
        total = sum(d for _, d in per_run)
        # The claim must be found AS a claim, not as a digit loose in the prose: every
        # digit from 1 to 10 appears somewhere in both files, so a substring test passes
        # whatever the records say. The first version of this test did exactly that —
        # the defect this file exists to catch, in the file that catches it.
        # `\s+`, not a literal space: the claim is a sentence, and a sentence gets
        # re-wrapped. The first version of this guard matched a single line only, so
        # re-flowing the paragraph would have silenced it without changing a word — the
        # dead-guard class round 3 filed three of.
        claim = re.compile(r"(\w+)\s+hallucinated\s+findings?\s+in\s+(\w+)")
        for where, text in (
            ("RESULTS.md", _results_text()),
            ("README.md", _readme_text()),
        ):
            found = claim.search(text)
            if offending == 0:
                assert not found, (
                    f"{where} claims '{found.group(0) if found else ''}' while no committed "
                    f"finding quotes text outside its manuscript across {total} findings"
                )
                continue
            # The >= 1 real match this binding needs: the regex is proven against the
            # real file, in both files, every time the records show a count to state.
            assert found, (
                f"{where} must state the measured count as a claim a reader can check: "
                f"{offending} of {total} reviewer findings quote text not in the "
                "manuscript, phrased as 'N hallucinated findings in M'"
            )
            assert (_as_number(found.group(1)), _as_number(found.group(2))) == (offending, total), (
                f"{where} says '{found.group(0)}'; the records say {offending} of {total}"
            )
        # The shape the prose actually uses ("0 of 32 reviewer findings quote text that is
        # not in the manuscript") went unbound while the regex above looked for a wording
        # neither file uses. It is bound with the rest of the population claims.
        _check_published_populations_recompute(_results_text(), _readme_text())


class TestTheseYieldBindingsCanGoRed:
    """A control per binding, over the REAL documents, mutated in memory.

    Round 4 wrote the rule these obey: a verification that cannot fail is worth nothing.
    So each control asserts the string it moves is present in the file as committed (a
    control over a string that is not there proves nothing), mutates it, and requires the
    checker to fail with the message that names the figure — not merely to fail, which any
    unrelated breakage would satisfy.
    """

    # Each row: (the file, the committed text, what an evaluator changed it to, the
    # message the binding must answer with). The first two rows are verbatim from round
    # 5's report 3, Finding 1 — the mutations that passed 594 green tests.
    MUTATIONS: ClassVar[list[tuple[str, str, str, str]]] = [
        (
            # Re-pointed when the README moved to the per-run lead (D-15). The row is
            # still round 5's mutation — the same figure moved by the same amount — but
            # the needle has to be text the page actually carries, or the control
            # mutates nothing and its green means nothing.
            "README.md",
            "12 of 40 verdicts counts the same 20 twice",
            "19 of 40 verdicts counts the same 20 twice",
            r"README\.md publishes 19 of 40",
        ),
        (
            "results/RESULTS.md",
            "**6 of 20 verdicts carry a surviving evidence span in each of the two "
            "committed runs**",
            "**9 of 20 verdicts carry a surviving evidence span in each of the two "
            "committed runs**",
            r"publishes 9 of 20",
        ),
        (
            "results/RESULTS.md",
            "0 of 16 reviewer findings quote text that is not in the manuscript",
            "3 of 16 reviewer findings quote text that is not in the manuscript",
            r"publishes 3 of 16",
        ),
        (
            "results/RESULTS.md",
            "6 of 20 verdicts change with the evidence order",
            "9 of 20 verdicts change with the evidence order",
            r"publishes 9 of 20",
        ),
    ]

    @pytest.mark.parametrize(("where", "original", "mutated", "message"), MUTATIONS)
    def test_moving_a_published_yield_fails(
        self, where: str, original: str, mutated: str, message: str
    ) -> None:
        results_text, readme = _results_text(), _readme_text()
        text = results_text if where.endswith("RESULTS.md") else readme
        assert original in text, (
            f"{where} no longer contains {original!r}, so this control moves a string that "
            "is not there and proves nothing. Re-point it at the sentence that carries the "
            "figure now."
        )
        if where.endswith("RESULTS.md"):
            results_text = results_text.replace(original, mutated, 1)
        else:
            readme = readme.replace(original, mutated, 1)
        with pytest.raises(AssertionError, match=message):
            _check_published_populations_recompute(results_text, readme)

    def test_moving_the_lens_count_fails(self) -> None:
        results_text, readme = _results_text(), _readme_text()
        original = "The deterministic lens: 0 findings"
        assert original in results_text, (
            f"results/RESULTS.md no longer contains {original!r}; re-point this control"
        )
        with pytest.raises(AssertionError, match=r"3 findings"):
            _check_the_lens_yield(
                results_text.replace(original, "The deterministic lens: 3 findings", 1), readme
            )

    def test_moving_the_reviewer_citation_count_fails(self) -> None:
        results_text, readme = _results_text(), _readme_text()
        original = "1 cites a retrieved chunk at all"
        assert original in results_text, (
            f"results/RESULTS.md no longer contains {original!r}; re-point this control"
        )
        with pytest.raises(AssertionError, match=r"reviewer-citation yield"):
            _check_the_reviewer_citation_yield(
                results_text.replace(original, "4 cites a retrieved chunk at all", 1), readme
            )

    def test_an_unclassifiable_verdict_population_fails(self) -> None:
        """A new sentence publishing "N of M verdicts" about nothing this test can
        recompute is an unbound number, and unbound is what Finding 1 is about."""
        readme = _readme_text() + "\n\nThe panel settled 7 of 20 verdicts on a Tuesday.\n"
        with pytest.raises(AssertionError, match=r"names no mechanism"):
            _check_published_populations_recompute(_results_text(), readme)

    def test_moving_the_verdict_distribution_fails(self) -> None:
        """The evaluator's own mutation, in the prose copy of the distribution."""
        results_text, readme = _results_text(), _readme_text()
        original = "0 SUPPORTS, 0 REFUTES, 20 abstentions per run"
        assert original in readme, (
            f"README.md no longer contains {original!r}; re-point this control at the "
            "sentence that carries the distribution now"
        )
        mutated = readme.replace(original, "2 SUPPORTS, 0 REFUTES, 18 abstentions per run", 1)
        with pytest.raises(AssertionError, match=r"publishes 2 SUPPORTS"):
            _check_the_verdict_distribution_in_prose(results_text, mutated)

    def test_the_doubled_rule_passes_on_a_compliant_document(self) -> None:
        """The other direction, and the one that matters while the prose is still wrong.

        A rule that is red on the documents it was written for has to be shown capable
        of green, or a prose fix cannot be told apart from a broken checker. RESULTS.md
        now takes the first road the rule names and states both figures per run, so the
        SECOND road is demonstrated here: the per-run sentences are written back as the
        doubled form with its decomposition attached, the README gains the sentence that
        names the reproduction, and the rule is required to pass on the result.
        """
        results_text, readme = _results_text(), _readme_text()
        edits = [
            (
                "README.md",
                "## What the panel measurably does, and does not do",
                "## What the panel measurably does, and does not do\n\nThe two committed "
                "runs are identical apart from wall-clock, so the second is a reproduction "
                "rather than an independent sample.",
            ),
            (
                "results/RESULTS.md",
                "In each committed run, **0 of 16 reviewer findings quote text that is "
                "not in the manuscript.**",
                "Across both runs, **0 of 32 reviewer findings quote text that is not in "
                "the manuscript** — 0 in each run.",
            ),
            (
                "results/RESULTS.md",
                "which now fires on 6 of 20 verdicts in each committed run where every "
                "run committed",
                "which now fires on 12 of 40 verdicts, 6 in each run, where every run "
                "committed",
            ),
        ]
        for where, original, replacement in edits:
            text = results_text if where.endswith("RESULTS.md") else readme
            assert original in text, (
                f"{where} no longer contains {original!r}, so this control demonstrates a "
                "fix to a sentence that is not there. Re-point it, or drop the row if the "
                "sentence has been restated per run already."
            )
            if where.endswith("RESULTS.md"):
                results_text = results_text.replace(original, replacement, 1)
            else:
                readme = readme.replace(original, replacement, 1)
        _check_a_doubled_population_says_so(results_text, readme)

    def test_a_doubled_population_with_no_caveat_fails(self) -> None:
        """And the red direction, on a document the rule currently passes."""
        results_text = _results_text()
        original = (
            "It fires now: **6 of 20 verdicts carry a surviving evidence span in each of "
            "the two committed runs**."
        )
        assert original in results_text, (
            f"results/RESULTS.md no longer contains {original!r}; re-point this control "
            "at the sentence that carries the grounding yield now"
        )
        stripped = results_text.replace(
            original,
            "It fires now: **12 of 40 verdicts carry a surviving evidence span**.",
            1,
        )
        with pytest.raises(AssertionError, match=r"one population counted twice"):
            _check_a_doubled_population_says_so(stripped, _readme_text())
