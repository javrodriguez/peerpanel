"""Planted-error generator, the three-arm harness and the baseline — no live models.

The harness tests drive the REAL `run_planted_eval` on the committed CI corpus with
stub providers: three arms are constructed, both counts are computed from the SAME
strings by the two rules, and the record carries the rule its headline was computed
under. Nothing here talks to Ollama or the network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from peerpanel.agents.reviewer_base import EXCERPT_WORDS, excerpt
from peerpanel.agents.schemas import CallStats
from peerpanel.evals.baseline import SCHEMA, run_baseline
from peerpanel.evals.planted import (
    ASSERTION_CUES,
    CLAUSE_BREAKS,
    DETECTION_RULE,
    NEGATION,
    SEED,
    PlantedError,
    asserts,
    detect,
    plant_errors,
)
from peerpanel.evals.planted_eval import (
    BASELINE_MODELS,
    BASELINE_SYSTEM,
    PANEL_SYSTEM,
    SCORING_NOTE,
    PlantedEvalReport,
    SystemResult,
    baseline_model_of,
    baseline_system,
    render_table,
    run_planted_eval,
    scored_residue,
)
from peerpanel.manuscripts.store import read_manuscript
from peerpanel.orchestration import PanelProviders
from peerpanel.providers.base import ChatResponse, TokenLedger
from peerpanel.text.chunks import sentences

SAMPLE = "\n".join(
    [
        "Introduction to the study of sulfur assimilation in budding yeast cells.",
        "We found that MET17 expression increased under sulfate limitation in all "
        "three biological replicates of the experiment reported here today.",
        "Growth differences reached significance with p = 0.003 across conditions.",
        "The mutant strain accounted for 47% of the observed colonies in the assay.",
        "A longer discussion paragraph follows here with more than twenty-five words in "
        "it so that the fabricated citation planting rule has somewhere sensible to "
        "attach its plausible looking but entirely invented reference string.",
    ]
)


class TestPlanting:
    def test_all_five_kinds_planted(self) -> None:
        planted = plant_errors(SAMPLE, "sample.txt")
        kinds = {e.kind for e in planted.errors}
        assert kinds == {
            "gene_symbol_swap",
            "effect_direction_flip",
            "impossible_pvalue",
            "fabricated_citation",
            "impossible_percentage",
        }

    def test_deterministic_for_a_given_seed(self) -> None:
        a = plant_errors(SAMPLE, "sample.txt")
        b = plant_errors(SAMPLE, "sample.txt")
        assert a.text == b.text
        assert [e.model_dump() for e in a.errors] == [e.model_dump() for e in b.errors]
        assert SEED == 20260831

    def test_the_text_actually_changed_and_records_the_span(self) -> None:
        planted = plant_errors(SAMPLE, "sample.txt")
        assert planted.text != SAMPLE
        for error in planted.errors:
            assert error.replacement in planted.text
            if error.original:
                assert error.original not in planted.text.splitlines()[error.line_no - 1]

    def test_direction_flip_reverses_the_claim(self) -> None:
        planted = plant_errors(SAMPLE, "sample.txt")
        flip = next(e for e in planted.errors if e.kind == "effect_direction_flip")
        assert (flip.original.lower(), flip.replacement) in (
            ("increased", "decreased"),
            ("decreased", "increased"),
            ("higher", "lower"),
            ("lower", "higher"),
        )

    def test_impossible_values_are_actually_impossible(self) -> None:
        planted = plant_errors(SAMPLE, "sample.txt")
        assert "p = 1.34" in planted.text  # a p-value above 1
        assert "412%" in planted.text  # a percentage above 100

    def test_empty_text_plants_nothing_without_crashing(self) -> None:
        planted = plant_errors("", "empty.txt")
        assert planted.errors == []


class TestDetection:
    def test_named_token_counts_as_a_catch(self) -> None:
        planted = plant_errors(SAMPLE, "sample.txt")
        pval = next(e for e in planted.errors if e.kind == "impossible_pvalue")
        assert detect(pval, ["The reported p = 1.34 is not a valid probability."])

    def test_described_but_unquoted_does_not_count(self) -> None:
        """The documented under-crediting: recall is a floor, not a ceiling."""
        planted = plant_errors(SAMPLE, "sample.txt")
        pval = next(e for e in planted.errors if e.kind == "impossible_pvalue")
        assert not detect(pval, ["The reported significance value is impossible."])

    def test_case_insensitive(self) -> None:
        planted = plant_errors(SAMPLE, "sample.txt")
        cite = next(e for e in planted.errors if e.kind == "fabricated_citation")
        assert detect(cite, ["reference to HOLLINGSWORTH cannot be located"])

    def test_no_findings_detects_nothing(self) -> None:
        planted = plant_errors(SAMPLE, "sample.txt")
        assert not any(detect(e, []) for e in planted.errors)


class _StubChat:
    name = "stub"

    def __init__(self, payload: str, tokens_per_call: int = 100) -> None:
        self.payload = payload
        self.tokens = tokens_per_call
        self.calls: list[float] = []

    def chat(
        self,
        *,
        system: str,
        user: str,
        json_schema: object = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        self.calls.append(temperature)
        return ChatResponse(
            text=self.payload,
            model="stub",
            prompt_tokens=self.tokens // 2,
            completion_tokens=self.tokens // 2,
        )


PAYLOAD = '{"findings": [{"text": "p value impossible", "quote": "p = 1.34"}]}'


class TestBaseline:
    def _retrieve(self, query: str) -> list[tuple[str, str]]:
        return [("d:0:x", "context " * 50)]

    def test_samples_until_the_panel_budget_is_matched(self) -> None:
        stub = _StubChat(PAYLOAD, tokens_per_call=100)
        result = run_baseline(
            manuscript_text=SAMPLE,
            retrieve=self._retrieve,
            title="t",
            provider=stub,
            target_tokens=450,
        )
        assert result.samples == 5  # stops once spend reaches the panel's
        assert result.total_tokens >= 450
        assert stub.calls[0] == 0.0  # deterministic core sample
        assert all(t == 0.7 for t in stub.calls[1:])  # self-consistency samples

    def test_findings_unioned_and_deduped(self) -> None:
        """The quote is collected for the record and NEVER scored: it is a verbatim
        slice of the manuscript the token was planted into, so appending it to the
        scored string credited an arm for reproducing the perturbed sentence."""
        stub = _StubChat(PAYLOAD)
        result = run_baseline(
            manuscript_text=SAMPLE,
            retrieve=self._retrieve,
            title="t",
            provider=stub,
            target_tokens=300,
        )
        assert "p = 1.34" in PAYLOAD  # the quote the stub returned
        assert result.finding_texts == ["p value impossible"]

    def test_max_samples_caps_a_cheap_model(self) -> None:
        stub = _StubChat(PAYLOAD, tokens_per_call=2)
        result = run_baseline(
            manuscript_text=SAMPLE,
            retrieve=self._retrieve,
            title="t",
            provider=stub,
            target_tokens=10_000,
            max_samples=3,
        )
        assert result.samples == 3

    def test_unparseable_sample_is_skipped_not_fatal(self) -> None:
        stub = _StubChat("{not json")
        result = run_baseline(
            manuscript_text=SAMPLE,
            retrieve=self._retrieve,
            title="t",
            provider=stub,
            target_tokens=150,
        )
        assert result.finding_texts == []
        assert result.samples >= 1

    def test_schema_caps_findings(self) -> None:
        assert SCHEMA["properties"]["findings"]["maxItems"] == 10  # type: ignore[index]


ROOT = Path(__file__).resolve().parents[1]
# The manuscript whose twin IS a CI corpus member, so the harness's comparability
# guard (an exclusion list that drops no chunks) is exercised rather than skipped.
SUBJECT = ROOT / "manuscripts" / "met17-auxotroph.txt"


class _NamedProvider:
    """Only the part of the protocol the arm-naming reads."""

    def __init__(self, name: str) -> None:
        self.name = name

    def chat(self, **kwargs: object) -> ChatResponse:  # pragma: no cover - never called
        raise AssertionError("this stub exists to be named, not called")


class TestTheThreeArms:
    """Requirement 7 asks for a pass rate per named local model, so each model is an
    arm of its own: a rate measured over a mixture belongs to neither model."""

    def test_one_arm_per_named_local_model(self) -> None:
        assert BASELINE_MODELS == ("llama3.1:8b", "qwen2:7b")
        names = [baseline_system(m) for m in BASELINE_MODELS]
        assert len(set(names)) == len(names), f"two arms would share a name: {names}"
        for model, name in zip(BASELINE_MODELS, names, strict=True):
            assert model in name, f"{name} does not name the model it measures"
            assert name.startswith(BASELINE_SYSTEM)
            assert name != PANEL_SYSTEM

    def test_the_arm_is_named_for_the_model_not_the_wire(self) -> None:
        """Both baselines ride the same wire, so a wire-named arm would publish two
        rows whose identity is equal — and the pass rates would be unattributable."""
        assert baseline_model_of(_NamedProvider("ollama-native:llama3.1:8b")) == "llama3.1:8b"
        assert baseline_model_of(_NamedProvider("ollama-openai:qwen2:7b")) == "qwen2:7b"
        assert baseline_model_of(_NamedProvider("stub")) == "stub"


def _stats() -> CallStats:
    """A CallStats from the ledger that produces it — never a literal, so a field
    added to the schema cannot be satisfied here by a stale hand-written stub."""
    return CallStats.from_ledger(TokenLedger())


def _arm(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "system": PANEL_SYSTEM,
        "detected": [],
        "named_token": [],
        "missed": [],
        "total_tokens": 0,
        "wall_s": 0.0,
        "detail": "",
        "excluded_docs": [],
        "dropped_chunks": 0,
        "finding_texts": [],
        "scored_texts": [],
        "quoted_sentences_dropped": 0,
        "model_calls": _stats(),
        "unparsed_calls": 0,
        "models": {},
    }
    return base | overrides


class TestTheRecordSchema:
    """Both counts and the rule are REQUIRED: a record that carries only the headline
    cannot be checked for the gap between assertion and restatement, and a number whose
    rule has to be inferred from a commit date is how a quotation counted as a detection
    for three published rounds."""

    def test_named_token_has_no_default(self) -> None:
        fields = dict(_arm())
        del fields["named_token"]
        with pytest.raises(ValidationError, match="named_token"):
            SystemResult(**fields)  # type: ignore[arg-type]

    def test_scored_texts_has_no_default(self) -> None:
        """The residue is what the counts were computed over. A record that carries only
        the unstripped `finding_texts` cannot be checked for the stripping, and round 4
        found the un-checkable version of exactly this: 63% of the scored strings were
        verbatim manuscript and nothing in the record said so."""
        fields = dict(_arm())
        del fields["scored_texts"]
        with pytest.raises(ValidationError, match="scored_texts"):
            SystemResult(**fields)  # type: ignore[arg-type]

    def test_quoted_sentences_dropped_has_no_default(self) -> None:
        fields = dict(_arm())
        del fields["quoted_sentences_dropped"]
        with pytest.raises(ValidationError, match="quoted_sentences_dropped"):
            SystemResult(**fields)  # type: ignore[arg-type]

    def test_detection_rule_has_no_default(self) -> None:
        with pytest.raises(ValidationError, match="detection_rule"):
            PlantedEvalReport(  # type: ignore[call-arg]
                manuscript="m.txt",
                twin_in_corpus=False,
                n_errors=0,
                error_kinds={},
                errors=[],
                skipped_kinds={},
                results=[],
                note="",
            )

    def test_render_table_prints_both_counts_for_every_arm(self) -> None:
        errors = [
            PlantedError(
                error_id="fabricated_citation-0",
                kind="fabricated_citation",
                original="",
                replacement="x",
                detection_token="Hollingsworth",
                line_no=1,
            )
        ]
        report = PlantedEvalReport(
            manuscript="m.txt",
            twin_in_corpus=False,
            n_errors=1,
            error_kinds={"fabricated_citation-0": "fabricated_citation"},
            errors=errors,
            skipped_kinds={},
            detection_rule=DETECTION_RULE,
            results=[
                SystemResult(**_arm(system=PANEL_SYSTEM)),  # type: ignore[arg-type]
                *[
                    SystemResult(
                        **_arm(  # type: ignore[arg-type]
                            system=baseline_system(model),
                            detected=["fabricated_citation-0"],
                            named_token=["fabricated_citation-0"],
                        )
                    )
                    for model in BASELINE_MODELS
                ],
            ],
            note="note",
        )
        table = render_table(report)
        assert "asserted" in table and "named token" in table
        assert DETECTION_RULE in table
        for model in BASELINE_MODELS:
            row = next(line for line in table.splitlines() if baseline_system(model) in line)
            assert "1/1" in row, row
        panel_row = next(
            line
            for line in table.splitlines()
            if line.strip().startswith(PANEL_SYSTEM) and "0/1" in line
        )
        assert panel_row.count("0/1") == 2, f"both counts must be printed: {panel_row}"

    def test_render_table_says_how_much_of_each_arm_was_quotation(self) -> None:
        """The captured `.log` is the surface a reader meets before the JSON, and an
        arm that scored nothing because it quoted everything earned its 0 differently
        from one that wrote its own prose and asserted nothing anyway."""
        report = PlantedEvalReport(
            manuscript="m.txt",
            twin_in_corpus=False,
            n_errors=0,
            error_kinds={},
            errors=[],
            skipped_kinds={},
            detection_rule=DETECTION_RULE,
            results=[
                SystemResult(  # type: ignore[arg-type]
                    **_arm(
                        finding_texts=["a quoted sentence.", "our own prose."],
                        scored_texts=["our own prose."],
                        quoted_sentences_dropped=1,
                    )
                )
            ],
            note="note",
        )
        assert "scored 1 of 2 strings after dropping 1 quoted sentence" in render_table(report)


class _PanelStub:
    """Schema-shaped nothing, so the REAL orchestration runs and the panel arm is a
    real arm — the same stub shape `test_panel_orchestration.py` drives run_panel with."""

    name = "stub-panel"

    def __init__(self) -> None:
        self.calls = 0

    def chat(
        self,
        *,
        system: str,
        user: str,
        json_schema: object = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        self.calls += 1
        schema = json.dumps(json_schema) if json_schema else ""
        if "scores" in schema:
            body = (
                '{"scores": {"soundness": 3, "presentation": 3, "contribution": 3},'
                ' "confidence": 3, "findings": []}'
            )
        elif "claims" in schema:
            body = '{"claims": []}'
        elif "verdict" in schema:
            body = '{"verdict": "NOT_ENOUGH_INFO", "evidence": []}'
        else:
            body = "no synthesis"
        return ChatResponse(text=body, model="stub", prompt_tokens=1, completion_tokens=1)


class _BaselineStub:
    """One finding that ASSERTS a planted defect and one that merely names another
    token. The two counts must then differ on the same strings — which is the whole
    point of publishing both."""

    def __init__(self, name: str, asserted_token: str, named_only_token: str) -> None:
        self.name = name
        self._payload = json.dumps(
            {
                "findings": [
                    {
                        "text": f"The reference {asserted_token} does not exist.",
                        "quote": "unscored",
                    },
                    {
                        "text": f"The excerpt mentions {named_only_token} in the methods.",
                        "quote": "unscored",
                    },
                ]
            }
        )

    def chat(
        self,
        *,
        system: str,
        user: str,
        json_schema: object = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        return ChatResponse(text=self._payload, model="stub", prompt_tokens=1, completion_tokens=1)


class _QuotingStub:
    """An arm whose whole finding is a sentence copied out of the manuscript it read.

    This is not a hypothetical shape: round 4 measured 147 of the 235 committed scored
    strings (63%) as verbatim slices of the perturbed manuscript, and every "named
    token" credit the two records carried was one of them. `scored_text` closed the
    `quote` route; this stub travels the OTHER one, `text`.
    """

    def __init__(self, name: str, quoted: str) -> None:
        self.name = name
        self._payload = json.dumps({"findings": [{"text": quoted, "quote": "unscored"}]})

    def chat(
        self,
        *,
        system: str,
        user: str,
        json_schema: object = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        return ChatResponse(text=self._payload, model="stub", prompt_tokens=1, completion_tokens=1)


class _EmbedStub:
    """A constant, non-zero query vector: the ranking is not what these tests measure,
    and a zero vector would make cosine similarity undefined."""

    name = "stub-embed"
    dim = 768

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        return np.full((len(texts), self.dim), 0.1, dtype=np.float32)


class _QuotingBaselineStub(_BaselineStub):
    """An arm that copies the manuscript instead of writing about it.

    This is what the real local models mostly do — 63% of the committed scored strings
    are pure quotation — so a harness test that only ever sees arms writing their own
    prose is not testing the path the records actually take.
    """

    def __init__(self, name: str, quoted_sentence: str) -> None:
        super().__init__(name, "unused", "unused")
        self.name = name
        self._payload = json.dumps(
            {"findings": [{"text": quoted_sentence, "quote": "unscored"}]}
        )


def _tokens() -> tuple[str, str]:
    """The tokens the harness will plant, derived the same way the harness derives
    them — never retyped, so a change to the planting rule cannot leave this test
    asserting against a token nothing plants."""
    _header, body = read_manuscript(SUBJECT)
    planted = plant_errors(body, SUBJECT.name, window_words=EXCERPT_WORDS)
    by_kind = {e.kind: e.detection_token for e in planted.errors}
    return by_kind["fabricated_citation"], by_kind["gene_symbol_swap"]


def _perturbed() -> str:
    """The perturbed body the harness plants and every arm is shown — derived here the
    same way the harness derives it, never retyped."""
    _header, body = read_manuscript(SUBJECT)
    return plant_errors(body, SUBJECT.name, window_words=EXCERPT_WORDS).text


def _a_manuscript_sentence() -> str:
    """One sentence of the perturbed excerpt that NAMES a planted token.

    Taken from what an arm can actually see (`excerpt`), so the stub quotes what a real
    arm could have quoted rather than text no model was shown; and carrying a token, so
    the string is one the upper bound would credit if it reached the scored surface —
    which is what makes the test below able to fail.
    """
    _header, body = read_manuscript(SUBJECT)
    planted = plant_errors(body, SUBJECT.name, window_words=EXCERPT_WORDS)
    tokens = [e.detection_token.lower() for e in planted.errors]
    return next(
        sentence
        for sentence in sentences(excerpt(planted.text))
        if len(sentence.split()) >= 12 and any(t in sentence.lower() for t in tokens)
    )


class TestQuotationNeverReachesTheScoredSurface:
    """D-11: `scored_text` drops the `quote` field, and until round 4 the repository
    published that as "only the finding's own prose is scored". It was not: nothing
    stopped a model putting manuscript text in `text`. The residue closes that route,
    and it can only ever lower a count — an arm that quotes and asserts nothing scores
    nothing, which is what this proves end to end through the real harness."""

    def _report(self) -> tuple[str, PlantedEvalReport]:
        quoted = _a_manuscript_sentence()
        panel = _PanelStub()
        report = run_planted_eval(
            ROOT,
            SUBJECT,
            PanelProviders(methods=panel, novelty=panel, verifier=panel, converger=panel),
            baseline_providers=[
                _QuotingStub(f"stub-wire:{model}", quoted) for model in BASELINE_MODELS
            ],
            embedder=_EmbedStub(),
            corpus="ci",
        )
        return quoted, report

    def test_a_finding_that_is_only_quotation_is_committed_but_not_scored(self) -> None:
        quoted, report = self._report()
        for model in BASELINE_MODELS:
            arm = next(r for r in report.results if r.system == baseline_system(model))
            assert arm.finding_texts == [quoted], (
                f"{arm.system}: the record must still commit what the arm wrote"
            )
            assert arm.scored_texts == [], (
                f"{arm.system}: verbatim manuscript reached the scored surface: "
                f"{arm.scored_texts}"
            )
            assert arm.quoted_sentences_dropped == len(sentences(quoted)), arm.system
            assert arm.detected == [] and arm.named_token == [], (
                f"{arm.system}: a pure quotation minted a credit "
                f"({arm.detected} / {arm.named_token})"
            )

    def test_the_same_quotation_would_have_been_credited_before_the_residue(self) -> None:
        """Non-vacuity: this control can go red. The quoted sentence names a planted
        token, so the OLD surface (the raw `finding_texts`) credits it under the upper
        bound — the credit is removed by the residue, not by the string being harmless."""
        quoted, report = self._report()
        credited_raw = [e.error_id for e in report.errors if detect(e, [quoted])]
        assert credited_raw, (
            "the quoted sentence names no planted token, so this test would pass even "
            "if the residue did nothing — pick a sentence carrying a planted token"
        )


class TestTheHarnessRunsThreeArms:
    """The real `run_planted_eval` on the committed CI corpus, stubs for the models."""

    def _report(self) -> PlantedEvalReport:
        asserted, named_only = _tokens()
        panel = _PanelStub()
        return run_planted_eval(
            ROOT,
            SUBJECT,
            PanelProviders(methods=panel, novelty=panel, verifier=panel, converger=panel),
            baseline_providers=[
                _BaselineStub(f"stub-wire:{model}", asserted, named_only)
                for model in BASELINE_MODELS
            ],
            embedder=_EmbedStub(),
            corpus="ci",
        )

    def _report_with_a_quoting_arm(self) -> PlantedEvalReport:
        """The same run, but one baseline arm writes a verbatim sentence of the body."""
        asserted, named_only = _tokens()
        quoted_sentence = next(
            line.strip()
            for line in _perturbed().splitlines()
            if len(line.split()) > 12
        )
        panel = _PanelStub()
        stubs = [_BaselineStub(f"stub-wire:{BASELINE_MODELS[0]}", asserted, named_only)]
        stubs.append(_QuotingBaselineStub(f"stub-wire:{BASELINE_MODELS[1]}", quoted_sentence))
        return run_planted_eval(
            ROOT,
            SUBJECT,
            PanelProviders(methods=panel, novelty=panel, verifier=panel, converger=panel),
            baseline_providers=stubs,
            embedder=_EmbedStub(),
            corpus="ci",
        )

    def test_the_exclusion_count_and_the_quotation_count_are_different_numbers(
        self,
    ) -> None:
        """Two counts, two names — the one that shipped as the other.

        `dropped_chunks` is what the arm's INDEX withheld; `quoted_sentences_dropped`
        is how much of the arm's own writing was copied out of the manuscript. They
        answer unrelated questions, and a regeneration published the second under the
        first because a local rebound the parameter. The conformance suite caught it on
        the committed records, an hour of model time after the mistake; this catches it
        in a second, before anything is re-run.

        The discriminating property is not that the two differ — they might coincide by
        luck — but that they vary differently. Every arm searches the SAME index, so the
        exclusion count is identical across arms; the quotation count is a property of
        what each arm wrote, so it is not. Under the bug the exclusion count varied per
        arm, which is the shape asserted against here.
        """
        report = self._report()
        by_arm = {a.system: a.dropped_chunks for a in report.results}
        assert len(set(by_arm.values())) == 1, (
            f"the arms report different exclusion counts {by_arm} while searching one "
            "index — dropped_chunks is carrying some per-arm quantity"
        )
        for arm in report.results:
            assert (arm.dropped_chunks > 0) == bool(arm.excluded_docs), (
                f"{arm.system}: dropped_chunks {arm.dropped_chunks} disagrees with "
                f"excluded_docs {arm.excluded_docs}"
            )
        # The stubs above assert rather than quote, so nothing would be stripped and the
        # quotation count would sit at 0 on every arm — a state in which this test could
        # not tell the two quantities apart at all. Re-run it with an arm that DOES quote,
        # using a real sentence out of the perturbed body every arm is shown.
        quoting = self._report_with_a_quoting_arm()
        by_arm = {a.system: a.dropped_chunks for a in quoting.results}
        assert len(set(by_arm.values())) == 1, (
            f"the arms report different exclusion counts {by_arm} once one of them "
            "quotes — dropped_chunks is carrying the quotation count"
        )
        quoted = [a.quoted_sentences_dropped for a in quoting.results]
        assert any(q > 0 for q in quoted), (
            "the quoting arm dropped nothing, so this test still cannot tell the two "
            f"counts apart: {quoted}"
        )

    def test_one_panel_row_and_one_row_per_baseline_model(self) -> None:
        report = self._report()
        assert [r.system for r in report.results] == [
            PANEL_SYSTEM,
            *[baseline_system(m) for m in BASELINE_MODELS],
        ]
        for model in BASELINE_MODELS:
            arm = next(r for r in report.results if r.system == baseline_system(model))
            # The record names the PROVIDER that served the arm, so a reader can tell
            # the wire from the record alone; the arm is named for the model.
            assert arm.models == {"single-agent": f"stub-wire:{model}"}

    def test_the_record_names_the_rule_its_headline_was_computed_under(self) -> None:
        assert self._report().detection_rule == DETECTION_RULE

    def test_the_headline_is_a_subset_of_its_upper_bound(self) -> None:
        report = self._report()
        for arm in report.results:
            assert set(arm.detected) <= set(arm.named_token), (
                f"{arm.system}: asserted {arm.detected} is not within named-token "
                f"{arm.named_token} — the headline escaped its own upper bound"
            )
            assert set(arm.missed) == {
                e.error_id for e in report.errors
            } - set(arm.detected), f"{arm.system}: missed is not the headline's complement"

    def test_both_counts_are_computed_from_the_same_strings_by_different_rules(self) -> None:
        """Non-vacuity: the gap between the counts is real on this run, so a headline
        silently computed by the substring rule would fail here."""
        report = self._report()
        arm = next(r for r in report.results if r.system == baseline_system(BASELINE_MODELS[0]))
        assert set(arm.detected) < set(arm.named_token), (
            "the stub asserted one planted defect and merely named another; the two "
            f"counts came back equal ({arm.detected} / {arm.named_token})"
        )
        for error in report.errors:
            assert detect(error, arm.finding_texts) == (error.error_id in arm.named_token)

    def test_the_record_commits_the_residue_its_counts_were_computed_over(self) -> None:
        """`scored_texts` is derivable from `finding_texts` by the one shared rule, so a
        reader can check the stripping instead of taking it on trust."""
        report = self._report()
        perturbed = _perturbed()
        for arm in report.results:
            expected, dropped = scored_residue(arm.finding_texts, perturbed)
            assert arm.scored_texts == expected, arm.system
            assert arm.quoted_sentences_dropped == dropped, arm.system
            for error in report.errors:
                assert asserts(error, arm.scored_texts) == (error.error_id in arm.detected)
                assert detect(error, arm.scored_texts) == (error.error_id in arm.named_token)

    def test_the_note_derives_a_budget_clause_per_baseline_arm(self) -> None:
        report = self._report()
        for model in BASELINE_MODELS:
            assert baseline_system(model) in report.note, (
                f"the note never mentions {model}'s arm, so its share is unattributable"
            )

    def test_a_record_with_no_baseline_arm_is_refused_before_any_model_runs(self) -> None:
        panel = _PanelStub()
        with pytest.raises(ValueError, match="at least one baseline arm"):
            run_planted_eval(
                ROOT,
                SUBJECT,
                PanelProviders(methods=panel, novelty=panel, verifier=panel, converger=panel),
                baseline_providers=[],
                embedder=_EmbedStub(),
                corpus="ci",
            )
        assert panel.calls == 0, "the refusal came after the panel had already run"

    def test_two_arms_that_would_share_a_name_are_refused(self) -> None:
        panel = _PanelStub()
        asserted, named_only = _tokens()
        twins = [_BaselineStub("wire-a:llama3.1:8b", asserted, named_only) for _ in range(2)]
        with pytest.raises(ValueError, match="same name"):
            run_planted_eval(
                ROOT,
                SUBJECT,
                PanelProviders(methods=panel, novelty=panel, verifier=panel, converger=panel),
                baseline_providers=list(twins),
                embedder=_EmbedStub(),
                corpus="ci",
            )
        assert panel.calls == 0


# --------------------------------------------------------------------------------------
# The note is pinned to the mechanism, not to a phrase.
#
# Round 4, unanimously (reports 1, 2 and 3): every committed record's `note` described the
# negation test as "a negation in the five words before the assertion cue disqualifying
# it" — the PRE-calibration mechanism, defeated by a control before the rule scored
# anything and replaced by clause scope. The wrong sentence survived 559 green tests
# because the only test that read `note` checked which arm names appeared in it.
#
# So this pins behaviour, not wording. The note has to SELECT a mechanism (clause scope,
# or a fixed word window), and the mechanism it selects — re-implemented here, from the
# note's own words, independently of `planted.py` — has to reach the same verdict as the
# shipped code on every control below. A rewrite that changes the sentence but not the
# rule stays green; a rewrite that describes a different rule goes red, and so does a
# change to the rule that leaves the sentence behind.
# --------------------------------------------------------------------------------------

# The token every control names, so a control can be credited at all.
_TOKEN = "Dcr2"
_CONTROL_ERROR = PlantedError(
    error_id="gene_symbol_swap-0",
    kind="gene_symbol_swap",
    original="Dcr1",
    replacement="Dcr2",
    detection_token=_TOKEN,
    line_no=1,
)

# The controls. The first is the sentence that DEFEATED the five-word window before any
# record was scored (DECISIONS D21); the second disagrees in the other direction; the
# rest are the negation shapes the two mechanisms are most likely to split on, plus one
# plain assertion both must credit.
_CONTROLS = (
    "Nothing about the Dcr2 reference is incorrect.",
    "Dcr2 is not shown; it is incorrect.",
    "The Dcr2 citation is incorrect.",
    "Dcr2, not Dcr1, is incorrect.",
    "No part of the Dcr2 annotation here is incorrect.",
    "Neither of the two Dcr2 mentions is incorrect.",
)

# The clause the records shipped, verbatim from `results/planted-eval-*.json` at commit
# b1f58dc. Kept as a literal rather than read from the records, because the records are
# regenerated and the point of this control is that the pin catches THIS text.
_SUPERSEDED_NOTE_CLAUSE = (
    "some sentence of the finding's own prose both names the planted token and asserts "
    "that something is wrong, with a negation in the five words before the assertion "
    "cue disqualifying it."
)

_FIXED_WINDOW = re.compile(
    r"\b(\w+)\s+words?\s+(?:before|back|preceding|in front of)\b|\b\w+-word\b", re.IGNORECASE
)
_CLAUSE_SCOPE = re.compile(r"\bclause\b", re.IGNORECASE)
_NUMBER_WORDS = {
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}


def _mechanism_the_note_describes(note: str) -> tuple[str, int | None]:
    """Which negation mechanism does this prose describe — clause scope, or an N-word
    window? A note that names both, or neither, describes nothing checkable."""
    window = _FIXED_WINDOW.search(note)
    clause = _CLAUSE_SCOPE.search(note)
    assert bool(window) != bool(clause), (
        "the note must name exactly one negation mechanism; it names "
        f"{'both' if window and clause else 'neither'}: {note}"
    )
    if clause:
        return "clause", None
    assert window is not None
    return "window", _NUMBER_WORDS.get((window.group(1) or "").lower(), 5)


def _credited(sentence: str, *, window: int | None) -> bool:
    """Score one control under a negation mechanism implemented HERE, not imported.

    The cue alternation and the negation vocabulary are the code's (they are not what
    drifted); the SCOPE is written out, because the scope is the thing the note has to
    describe. `window=None` is clause scope — back to the sentence start or the last
    clause break; `window=n` is a fixed n-word lookback.
    """
    if _TOKEN.lower() not in sentence.lower():
        return False
    for cue in ASSERTION_CUES.finditer(sentence):
        before = sentence[: cue.start()]
        if window is None:
            breaks = list(CLAUSE_BREAKS.finditer(before))
            if breaks:
                before = before[breaks[-1].end() :]
        else:
            before = " ".join(before.split()[-window:])
        if not NEGATION.search(before):
            return True
    return False


def _assert_the_note_pins_the_mechanism(note: str) -> None:
    """Every check this pin makes, over any candidate note — so the control below can
    run the identical pin over the sentence that shipped and show it failing."""
    _kind, window = _mechanism_the_note_describes(note)
    disagreements = [
        sentence
        for sentence in _CONTROLS
        if _credited(sentence, window=window) != asserts(_CONTROL_ERROR, [sentence])
    ]
    assert not disagreements, (
        "the rule this note describes and the rule that runs reach different verdicts "
        f"on {disagreements} — the record would misstate how its own number was made"
    )
    for word in re.findall(r"[a-z]{3,}", CLAUSE_BREAKS.pattern):
        assert word in note, (
            f"the code breaks a clause at {word!r} and the note never says so — a reader "
            "re-scoring by hand would scan a different span"
        )
    for marker, limit in (
        ("real assertion", "that a negation can suppress a REAL assertion"),
        ("same-sentence only", "that the rule is same-sentence only"),
    ):
        assert marker.lower() in note.lower(), (
            f"the note drops a limit the rule has and `asserts` discloses: {limit}"
        )


class TestTheNoteDescribesTheRuleThatRan:
    def test_the_shipped_note_is_pinned_to_the_shipped_mechanism(self) -> None:
        _assert_the_note_pins_the_mechanism(SCORING_NOTE)

    def test_the_disclosed_limits_are_the_code_s_actual_behaviour(self) -> None:
        """Disclosing a limit the rule does not have would be its own drift, so each
        limit the note states is exercised against the code."""
        assert not asserts(_CONTROL_ERROR, ["Dcr2, not Dcr1, is incorrect."]), (
            "the note says a negation can suppress a real assertion; the code credited one"
        )
        assert not asserts(_CONTROL_ERROR, ["Dcr2 is named here. That symbol is incorrect."]), (
            "the note says the rule is same-sentence only; the code crossed a boundary"
        )

    def test_the_two_mechanisms_genuinely_disagree_so_this_pin_can_fail(self) -> None:
        """A control that cannot go red is worth nothing. The clause-scoped and
        five-word readings must disagree on at least one control, and the code must
        side with clause scope on all of them."""
        split = [c for c in _CONTROLS if _credited(c, window=None) != _credited(c, window=5)]
        assert "Nothing about the Dcr2 reference is incorrect." in split, split
        for sentence in _CONTROLS:
            assert _credited(sentence, window=None) == asserts(_CONTROL_ERROR, [sentence]), (
                f"the clause-scoped reading and the shipped code disagree on {sentence!r}"
            )

    def test_the_sentence_that_shipped_fails_this_pin(self) -> None:
        """The proof this test would have caught round 4's finding: the same pin, run
        over the clause the two committed records carried."""
        with pytest.raises(AssertionError, match=r"different verdicts"):
            _assert_the_note_pins_the_mechanism(_SUPERSEDED_NOTE_CLAUSE)
        assert _mechanism_the_note_describes(_SUPERSEDED_NOTE_CLAUSE) == ("window", 5)

    def test_the_note_never_reads_as_a_fixed_window(self) -> None:
        assert not _FIXED_WINDOW.search(SCORING_NOTE), (
            "the note describes a fixed word window; the code scopes by clause"
        )
        assert "five words" not in SCORING_NOTE.lower()
