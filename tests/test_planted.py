"""Planted-error generator, the three-arm harness and the baseline — no live models.

The harness tests drive the REAL `run_planted_eval` on the committed CI corpus with
stub providers: three arms are constructed, both counts are computed from the SAME
strings by the two rules, and the record carries the rule its headline was computed
under. Nothing here talks to Ollama or the network.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from peerpanel.agents.reviewer_base import EXCERPT_WORDS
from peerpanel.agents.schemas import CallStats
from peerpanel.evals.baseline import SCHEMA, run_baseline
from peerpanel.evals.planted import DETECTION_RULE, SEED, PlantedError, detect, plant_errors
from peerpanel.evals.planted_eval import (
    BASELINE_MODELS,
    BASELINE_SYSTEM,
    PANEL_SYSTEM,
    PlantedEvalReport,
    SystemResult,
    baseline_model_of,
    baseline_system,
    render_table,
    run_planted_eval,
)
from peerpanel.manuscripts.store import read_manuscript
from peerpanel.orchestration import PanelProviders
from peerpanel.providers.base import ChatResponse, TokenLedger

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


class _EmbedStub:
    """A constant, non-zero query vector: the ranking is not what these tests measure,
    and a zero vector would make cosine similarity undefined."""

    name = "stub-embed"
    dim = 768

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        return np.full((len(texts), self.dim), 0.1, dtype=np.float32)


def _tokens() -> tuple[str, str]:
    """The tokens the harness will plant, derived the same way the harness derives
    them — never retyped, so a change to the planting rule cannot leave this test
    asserting against a token nothing plants."""
    _header, body = read_manuscript(SUBJECT)
    planted = plant_errors(body, SUBJECT.name, window_words=EXCERPT_WORDS)
    by_kind = {e.kind: e.detection_token for e in planted.errors}
    return by_kind["fabricated_citation"], by_kind["gene_symbol_swap"]


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
