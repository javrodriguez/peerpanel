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

import pytest

from peerpanel.agents.claims_verifier import (
    MIN_SWAP_N,
    SOURCE_MANUSCRIPT,
    claim_source,
    swap_consistency_by_source,
    swap_consistency_rate,
)
from peerpanel.agents.reviewer_base import MAX_FINDINGS
from peerpanel.agents.schemas import (
    RUBRIC_DIMENSIONS,
    VERDICT_NEI,
    VERDICT_REFUTES,
    VERDICTS,
    CallStats,
    PanelReview,
)
from peerpanel.embeddings.query_cache import CachedQueryEmbedder
from peerpanel.evals.ablation import run_ablation
from peerpanel.evals.planted import detect
from peerpanel.evals.planted_eval import BASELINE_SYSTEM, PANEL_SYSTEM, PlantedEvalReport
from peerpanel.graph.extract import EXTRACTION_PROVIDER_NAME
from peerpanel.orchestration.panel import detect_conflicts

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PANEL_RECORDS = sorted(p.name for p in RESULTS.glob("panel-review-*.json"))
PLANTED_RECORDS = sorted(p.name for p in RESULTS.glob("planted-eval-*.json"))


def _panel(name: str) -> PanelReview:
    return PanelReview.model_validate_json((RESULTS / name).read_text())


def _planted(name: str) -> PlantedEvalReport:
    return PlantedEvalReport.model_validate_json((RESULTS / name).read_text())


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

    def test_the_two_arms_are_the_two_systems_the_code_names(self, name: str) -> None:
        assert {arm.system for arm in _planted(name).results} == {PANEL_SYSTEM, BASELINE_SYSTEM}

    def test_detection_recomputes_from_the_scored_texts(self, name: str) -> None:
        """Both arms commit the exact strings they were scored on. Re-running the
        detector over them must reproduce detected/missed to the id."""
        report = _planted(name)
        ids = [e.error_id for e in report.errors]
        assert report.n_errors == len(ids)
        assert report.error_kinds == {e.error_id: e.kind for e in report.errors}
        for arm in report.results:
            detected = [e.error_id for e in report.errors if detect(e, arm.finding_texts)]
            missed = [e.error_id for e in report.errors if not detect(e, arm.finding_texts)]
            assert (arm.detected, arm.missed) == (detected, missed), arm.system
            assert set(arm.detected) | set(arm.missed) == set(ids), arm.system
            assert not set(arm.detected) & set(arm.missed), arm.system

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

    def test_both_arms_report_the_same_exclusion(self, name: str) -> None:
        states = {(tuple(a.excluded_docs), a.dropped_chunks) for a in _planted(name).results}
        assert len(states) == 1, f"the two arms searched different indexes: {states}"


class TestAblationRecordIsThisCodesOutput:
    def test_ci_ladder_regenerates_byte_for_byte_except_latency(self) -> None:
        """The committed CI ablation is a deterministic function of committed bytes
        (corpus, extractions, query vectors). Regenerate it here and compare
        everything but wall-clock: any other difference means the record was not
        produced by this code."""
        committed = json.loads((RESULTS / "ablation-ci.json").read_text())
        fresh = run_ablation(ROOT, CachedQueryEmbedder(ROOT), corpus="ci").model_dump()

        def strip(report: dict[str, object]) -> dict[str, object]:
            rows = report["results"]
            assert isinstance(rows, list)
            return {
                **report,
                "results": [{k: v for k, v in row.items() if k != "latency_ms"} for row in rows],
            }

        assert strip(fresh) == strip(committed)


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


def _read_whole(stats: CallStats, where: str, *, calls_required: bool = True) -> None:
    """The D19 proof: every prompt this record's calls sent fit the window it ran in.

    Read per call, not across the record: the largest prompt and the smallest window
    usually belong to different calls, so comparing those two refuses honest records
    (a 4,100-token reviewer prompt in an 8,192 window beside a short converger call in
    a 4,096 one) while proving nothing about either.
    """
    if stats.calls == 0:
        # A rebuild that replayed a complete cache called no model. It may not claim
        # measurements it never made — that would be a mock in the numbers.
        assert not calls_required, f"{where}: a record with no model calls measured nothing"
        assert stats.largest_prompt_tokens == 0, f"{where}: no calls, but a prompt measured"
        assert stats.smallest_context is None and stats.smallest_headroom_tokens is None, where
        return
    assert stats.largest_prompt_tokens > 0, where
    assert stats.smallest_context is not None, f"{where}: a wire that reports no window"
    assert stats.smallest_headroom_tokens is not None, f"{where}: no per-call margin recorded"
    assert stats.smallest_headroom_tokens > 0, (
        f"{where}: a prompt came within {stats.smallest_headroom_tokens} tokens of the window "
        f"it ran in (largest prompt {stats.largest_prompt_tokens:,}, smallest window "
        f"{stats.smallest_context:,}); a prompt was cut"
    )


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
    def test_both_arms_of_the_planted_evaluation(self, name: str) -> None:
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
        _read_whole(stats, name, calls_required=False)
        if stats.calls == 0:
            # Nothing was extracted, so every chunk must already have been on disk. Only
            # this direction is sound: `cached_before` counts FILES in the cache directory
            # (pipeline.py) while a hit is keyed by chunk id, provider and PROMPT_VERSION,
            # so orphans and a bumped prompt version both inflate it above the number of
            # usable records — an honest cold build can show cached_before == chunks.
            assert record["cached_before"] >= record["chunks"], (
                f"{name}: no model calls, but only {record['cached_before']} cached "
                f"records for {record['chunks']} chunks"
            )
        assert record["truncated_chunks"] == 0, f"{name}: a chunk recorded as truncated"


class TestClaimedYieldsMatchTheRecords:
    """Two capabilities are described in prose as strengths. Their measured yield
    is bound here so the description can never drift from the artifacts again."""

    def _panel_runs(self) -> list[PanelReview]:
        return [_panel(n) for n in PANEL_RECORDS]

    def test_grounding_yield_is_stated_accurately(self) -> None:
        grounded = sum(1 for r in self._panel_runs() for v in r.verdicts if v.evidence)
        total = sum(len(r.verdicts) for r in self._panel_runs())
        results_text = (RESULTS / "RESULTS.md").read_text()
        readme = (ROOT / "README.md").read_text()
        if grounded == 0:
            for text, where in ((results_text, "RESULTS.md"), (readme, "README.md")):
                assert "never" in text.lower() and "zero" in text.lower(), (
                    f"{where} must say the grounding yield is zero while it is "
                    f"({grounded}/{total} spans across committed runs)"
                )
        assert f"{total}" in results_text, "the verdict population should be published"

    def test_deterministic_lens_yield(self) -> None:
        found = sum(len(r.deterministic_findings) for r in self._panel_runs())
        if found == 0:
            assert "0 findings" in (RESULTS / "RESULTS.md").read_text()

    def test_the_hallucination_count_is_the_records_own_count(self) -> None:
        """The one number the prose reports AGAINST itself, and the only one that was
        published without a binding: reviewer findings quoting text that is not in the
        manuscript under review. Recomputed here from the records, both files required to
        state it, so the count can never be softened by a rewrite or left behind by a
        re-run."""
        offending = total = 0
        for name in PANEL_RECORDS:
            review = _panel(name)
            # Two passes on purpose: "-run\d+$" cannot match while ".json" still trails,
            # so a single alternation leaves "met17-auxotroph-run2" and looks for a
            # manuscript by that name.
            stem = re.sub(r"-run\d+$", "", Path(name).stem.replace("panel-review-", "", 1))
            manuscript = _normalise((ROOT / "manuscripts" / f"{stem}.txt").read_text())
            for output in review.reviewer_outputs:
                for finding in output.findings:
                    total += 1
                    if finding.quote and _normalise(finding.quote) not in manuscript:
                        offending += 1
        # The claim must be found AS a claim, not as a digit loose in the prose: every
        # digit from 1 to 10 appears somewhere in both files, so a substring test passes
        # whatever the records say. The first version of this test did exactly that —
        # the defect this file exists to catch, in the file that catches it.
        claim = re.compile(r"(\w+) hallucinated findings? in (\w+)")
        for where, text in (
            ("RESULTS.md", (RESULTS / "RESULTS.md").read_text()),
            ("README.md", (ROOT / "README.md").read_text()),
        ):
            found = claim.search(text)
            if offending == 0:
                assert not found, (
                    f"{where} claims '{found.group(0) if found else ''}' while no committed "
                    f"finding quotes text outside its manuscript across {total} findings"
                )
                continue
            assert found, (
                f"{where} must state the measured count as a claim a reader can check: "
                f"{offending} of {total} reviewer findings quote text not in the "
                "manuscript, phrased as 'N hallucinated findings in M'"
            )
            assert (_as_number(found.group(1)), _as_number(found.group(2))) == (offending, total), (
                f"{where} says '{found.group(0)}'; the records say {offending} of {total}"
            )
