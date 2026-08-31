"""Planted-error generator + baseline tests — deterministic, no model calls."""

from __future__ import annotations

from peerpanel.evals.baseline import SCHEMA, run_baseline
from peerpanel.evals.planted import SEED, detect, plant_errors
from peerpanel.providers.base import ChatResponse

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

    def chat(self, *, system: str, user: str, json_schema: object = None,
             temperature: float = 0.0, max_tokens: int = 2048) -> ChatResponse:
        self.calls.append(temperature)
        return ChatResponse(
            text=self.payload, model="stub",
            prompt_tokens=self.tokens // 2, completion_tokens=self.tokens // 2,
        )


PAYLOAD = '{"findings": [{"text": "p value impossible", "quote": "p = 1.34"}]}'


class TestBaseline:
    def _retrieve(self, query: str) -> list[tuple[str, str]]:
        return [("d:0:x", "context " * 50)]

    def test_samples_until_the_panel_budget_is_matched(self) -> None:
        stub = _StubChat(PAYLOAD, tokens_per_call=100)
        result = run_baseline(
            manuscript_text=SAMPLE, retrieve=self._retrieve, title="t",
            provider=stub, target_tokens=450,
        )
        assert result.samples == 5  # stops once spend reaches the panel's
        assert result.total_tokens >= 450
        assert stub.calls[0] == 0.0  # deterministic core sample
        assert all(t == 0.7 for t in stub.calls[1:])  # self-consistency samples

    def test_findings_unioned_and_deduped(self) -> None:
        stub = _StubChat(PAYLOAD)
        result = run_baseline(
            manuscript_text=SAMPLE, retrieve=self._retrieve, title="t",
            provider=stub, target_tokens=300,
        )
        assert result.finding_texts == ["p value impossible p = 1.34"]

    def test_max_samples_caps_a_cheap_model(self) -> None:
        stub = _StubChat(PAYLOAD, tokens_per_call=2)
        result = run_baseline(
            manuscript_text=SAMPLE, retrieve=self._retrieve, title="t",
            provider=stub, target_tokens=10_000, max_samples=3,
        )
        assert result.samples == 3

    def test_unparseable_sample_is_skipped_not_fatal(self) -> None:
        stub = _StubChat("{not json")
        result = run_baseline(
            manuscript_text=SAMPLE, retrieve=self._retrieve, title="t",
            provider=stub, target_tokens=150,
        )
        assert result.finding_texts == []
        assert result.samples >= 1

    def test_schema_caps_findings(self) -> None:
        assert SCHEMA["properties"]["findings"]["maxItems"] == 10  # type: ignore[index]
