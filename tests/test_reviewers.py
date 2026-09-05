"""Reviewer tests — grounded prompts, hygiene, structural query derivation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from peerpanel.agents.methods_reviewer import methods_queries
from peerpanel.agents.novelty_reviewer import novelty_queries
from peerpanel.agents.reviewer_base import (
    EXCERPT_WORDS,
    MAX_FINDINGS,
    REVIEW_SCHEMA,
    SYSTEM_TEMPLATE,
    excerpt,
    run_reviewer,
)
from peerpanel.agents.schemas import ReviewerOutput
from peerpanel.providers.base import ChatResponse, TokenLedger

# "You are a rigorous / expert / senior reviewer" is the persona shape the README
# says this project does not use; "You are reviewing a manuscript" is a task.
PERSONA = re.compile(r"\byou are (a|an|the)\b", re.IGNORECASE)
# The words live in data beside the denied claims: held here as prose, one of
# them is itself a denied claim and the sweep trips on the test that bans it.
PERSONA_WORDS: tuple[str, ...] = tuple(
    json.loads((Path(__file__).parent / "denied_claims.json").read_text())["persona_words"]
)


class _StubChat:
    name = "stub-chat"

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        *,
        system: str,
        user: str,
        json_schema: object = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        self.calls.append({"system": system, "user": user, "schema": json_schema})
        return ChatResponse(text=self.payload, model="stub", prompt_tokens=7, completion_tokens=3)


GOOD = (
    '{"scores": {"soundness": 4, "presentation": 3, "contribution": 4}, "confidence": 3,'
    ' "findings": [{"dimension": "soundness", "severity": "major",'
    ' "text": "No replication stated.", "quote": "n = 1 culture",'
    ' "evidence_chunk_ids": ["docA:0:aaaa0000"]},'
    ' {"dimension": "contribution", "severity": "minor",'
    ' "text": "Overlaps known work.", "evidence_chunk_ids": ["docZ:9:ffff9999"]}]}'
)


def _retrieve(query: str) -> list[tuple[str, str]]:
    return [("docA:0:aaaa0000", "Met4 activates sulfur metabolism " * 30)]


# The one chunk `_retrieve` provides, and an id it never does: the whole hygiene
# question is which of the two a finding cited.
RETRIEVED = "docA:0:aaaa0000"
NEVER_RETRIEVED = "docZ:9:ffff9999"


def _payload(findings: list[object]) -> str:
    return json.dumps(
        {
            "scores": {"soundness": 3, "presentation": 3, "contribution": 3},
            "confidence": 3,
            "findings": findings,
        }
    )


def _a_finding(**over: object) -> dict[str, object]:
    """A finding the hygiene pass keeps whole, unless a test breaks one field of it."""
    base: dict[str, object] = {
        "dimension": "soundness",
        "severity": "minor",
        "text": "A finding.",
        "quote": "",
        "evidence_chunk_ids": [],
    }
    base.update(over)
    return base


def _review(payload: str) -> ReviewerOutput:
    return run_reviewer(
        reviewer_name="test-lens",
        provider=_StubChat(payload),  # type: ignore[arg-type]
        manuscript_text="word " * 2000,
        retrieve=_retrieve,
        focus="testing",
        queries=["q1"],
    )


class TestRunReviewer:
    def _run(self, payload: str = GOOD) -> tuple[object, _StubChat, TokenLedger]:
        stub = _StubChat(payload)
        ledger = TokenLedger()
        out = run_reviewer(
            reviewer_name="test-lens",
            provider=stub,  # type: ignore[arg-type]
            manuscript_text="word " * 2000,
            retrieve=_retrieve,
            focus="testing",
            queries=["q1"],
            ledger=ledger,
        )
        return out, stub, ledger

    def test_prompt_carries_excerpt_context_and_schema(self) -> None:
        _, stub, _ = self._run()
        call = stub.calls[0]
        assert "[docA:0:aaaa0000]" in str(call["user"])
        assert call["schema"] == REVIEW_SCHEMA
        assert len(str(call["user"]).split()) < 1400  # slot-context budget respected

    def test_no_persona_in_the_system_prompt(self) -> None:
        """The README's claim that reviewers differ structurally, not by persona,
        is held here as a shape the prompt must not take — an earlier version of
        this test only checked the word "persona" was absent, which no prompt
        would ever contain."""
        _, stub, _ = self._run()
        system = str(stub.calls[0]["system"])
        assert not PERSONA.search(system), system
        assert not any(w in system.lower() for w in PERSONA_WORDS), system
        assert SYSTEM_TEMPLATE.startswith("You are reviewing")  # a task, not an identity

    def test_quote_is_asked_for_required_and_kept(self) -> None:
        """The quote channel is what the planted-error scorer reads; the baseline
        gets the same ask, so it must be in the reviewer's prompt AND schema."""
        out, stub, _ = self._run()
        assert "quote" in str(stub.calls[0]["system"])
        item = REVIEW_SCHEMA["properties"]["findings"]["items"]
        assert "quote" in item["required"]
        findings = out.findings  # type: ignore[attr-defined]
        assert findings[0].quote == "n = 1 culture"
        assert findings[1].quote == ""  # absent quote is empty, never a crash

    def test_hygiene_drops_ungrounded_evidence_ids(self) -> None:
        out, _, _ = self._run()
        findings = out.findings  # type: ignore[attr-defined]
        assert findings[0].evidence_chunk_ids == ["docA:0:aaaa0000"]
        assert findings[1].evidence_chunk_ids == []  # docZ never provided

    def test_scores_clamped_and_ledger_recorded(self) -> None:
        out, _, ledger = self._run(
            '{"scores": {"soundness": 9, "presentation": -2, "contribution": 3},'
            ' "confidence": 11, "findings": []}'
        )
        assert out.scores == {"soundness": 5, "presentation": 1, "contribution": 3}  # type: ignore[attr-defined]
        assert out.confidence == 5  # type: ignore[attr-defined]
        assert ledger.calls == 1 and ledger.total_tokens == 10

    def test_invalid_json_is_honest(self) -> None:
        out, _, _ = self._run('{"scor')
        assert out.truncated  # type: ignore[attr-defined]
        assert out.findings == []  # type: ignore[attr-defined]

    def test_schema_violating_but_parseable_json_does_not_crash(self) -> None:
        """Constrained decoding binds on ONE wire; every other wire can return
        JSON that parses and violates the schema. A reviewer that raised here
        would take the whole panel down with it."""
        payload = json.dumps(
            {
                "scores": {"soundness": "four", "presentation": None, "contribution": 2.6},
                "confidence": "high",
                "findings": [
                    {
                        "dimension": "soundness",
                        "text": "no severity given",
                        "evidence_chunk_ids": "docA:0:aaaa0000",
                    },  # a string, not a list
                    {
                        "dimension": "presentation",
                        "severity": "catastrophic",
                        "text": "bad severity",
                        "quote": None,
                        "evidence_chunk_ids": [],
                    },
                    {"dimension": "vibes", "severity": "major", "text": "bad dimension"},
                    "not even an object",
                    {"dimension": "soundness", "severity": "major", "text": "   "},
                ],
            }
        )
        out, _, _ = self._run(payload)
        assert not out.truncated  # type: ignore[attr-defined]
        assert out.scores == {"soundness": 1, "presentation": 1, "contribution": 2}  # type: ignore[attr-defined]
        assert out.confidence == 1  # type: ignore[attr-defined]
        findings = out.findings  # type: ignore[attr-defined]
        assert [f.text for f in findings] == ["no severity given", "bad severity"]
        assert [f.severity for f in findings] == ["minor", "minor"]
        assert findings[0].evidence_chunk_ids == []
        assert findings[1].quote == ""

    def test_findings_are_capped_in_code_not_by_the_schema(self) -> None:
        many = [
            {
                "dimension": "soundness",
                "severity": "minor",
                "text": f"f{i}",
                "quote": "",
                "evidence_chunk_ids": [],
            }
            for i in range(MAX_FINDINGS + 5)
        ]
        payload = json.dumps({"scores": {}, "confidence": 3, "findings": many})
        out, _, _ = self._run(payload)
        assert len(out.findings) == MAX_FINDINGS  # type: ignore[attr-defined]
        assert out.findings[0].text == "f0"  # type: ignore[attr-defined]

    def test_a_non_object_payload_is_truncated_not_a_crash(self) -> None:
        out, _, _ = self._run("[1, 2, 3]")
        assert out.truncated  # type: ignore[attr-defined]


class TestHygieneCounters:
    """What the hygiene pass removed, on the record, per reviewer.

    Round 4 finding 9: `results/RESULTS.md` publishes "of 16 findings in a run, 1
    cites a retrieved chunk at all" and reads it as a fact about the reviewers, while
    the pass that produced that 1 deleted citations to unprovided chunks and discarded
    whole findings without counting either. From such a record it is impossible to
    tell a reviewer that cited nothing from one whose citations this pass deleted.
    These tests hold both counters to what the filter actually did.
    """

    def test_a_citation_to_a_chunk_never_retrieved_is_counted_and_the_finding_kept(
        self,
    ) -> None:
        out = _review(_payload([_a_finding(evidence_chunk_ids=[NEVER_RETRIEVED])]))
        assert out.unretrieved_citations_dropped == 1
        assert out.malformed_findings_dropped == 0
        # The finding survives — which is exactly why the drop needs a counter.
        assert [f.text for f in out.findings] == ["A finding."]
        assert out.findings[0].evidence_chunk_ids == []

    def test_every_removed_id_is_counted_not_every_finding(self) -> None:
        """Two unretrieved ids in one finding are two removals: the count is of
        citations, so a finding citing three ghosts cannot read as one."""
        out = _review(
            _payload(
                [_a_finding(evidence_chunk_ids=[NEVER_RETRIEVED, RETRIEVED, "docQ:1:0000ffff"])]
            )
        )
        assert out.unretrieved_citations_dropped == 2
        assert out.findings[0].evidence_chunk_ids == [RETRIEVED]

    def test_a_bad_dimension_an_empty_text_and_a_non_object_are_each_a_dropped_finding(
        self,
    ) -> None:
        out = _review(
            _payload(
                [
                    _a_finding(dimension="vibes"),
                    _a_finding(text="   "),
                    "not even an object",
                    _a_finding(text="The one that survives."),
                ]
            )
        )
        assert out.malformed_findings_dropped == 3
        assert out.unretrieved_citations_dropped == 0
        assert [f.text for f in out.findings] == ["The one that survives."]

    def test_a_clean_run_counts_no_drops(self) -> None:
        out = _review(
            _payload(
                [
                    _a_finding(evidence_chunk_ids=[RETRIEVED]),
                    _a_finding(dimension="presentation", text="Second.", evidence_chunk_ids=[]),
                ]
            )
        )
        assert out.unretrieved_citations_dropped == 0
        assert out.malformed_findings_dropped == 0
        assert len(out.findings) == 2

    def test_a_citation_field_that_is_not_a_list_counts_as_one_removal(self) -> None:
        """Constrained decoding binds on one wire; another can hand back a bare
        string where the ids belong. That field is discarded whole, and counting it
        0 would recreate the silent drop these counters exist to end. An ABSENT
        field is not a removal — nothing was cited."""
        bare_string = _review(_payload([_a_finding(evidence_chunk_ids=RETRIEVED)]))
        assert bare_string.unretrieved_citations_dropped == 1
        assert bare_string.findings[0].evidence_chunk_ids == []

        absent = _a_finding()
        del absent["evidence_chunk_ids"]
        assert _review(_payload([absent])).unretrieved_citations_dropped == 0

    def test_a_deleted_citation_is_visible_where_the_findings_are_identical(self) -> None:
        """THE CONTROL. Two runs whose findings are identical — one reviewer cited
        nothing, the other cited a chunk it was never given — must not produce the
        same record. A counter hardcoded to 0, or one derived from the citations that
        SURVIVED, is red here and green everywhere a fixed expectation is asserted."""
        cited_nothing = _review(_payload([_a_finding(evidence_chunk_ids=[])]))
        citation_deleted = _review(_payload([_a_finding(evidence_chunk_ids=[NEVER_RETRIEVED])]))
        assert cited_nothing.findings == citation_deleted.findings  # indistinguishable there
        assert cited_nothing.unretrieved_citations_dropped == 0
        assert citation_deleted.unretrieved_citations_dropped == 1
        assert cited_nothing != citation_deleted  # and distinguishable in the record

    def test_a_discarded_finding_is_visible_where_the_findings_are_identical(self) -> None:
        """The same control for the other drop: a reviewer that wrote one finding and
        one that wrote two, one of them unusable, carry the same `findings` list."""
        wrote_one = _review(_payload([_a_finding()]))
        one_discarded = _review(_payload([_a_finding(), _a_finding(dimension="vibes")]))
        assert wrote_one.findings == one_discarded.findings
        assert wrote_one.malformed_findings_dropped == 0
        assert one_discarded.malformed_findings_dropped == 1
        assert wrote_one != one_discarded

    def test_the_counters_describe_only_the_findings_the_record_carries(self) -> None:
        """Past MAX_FINDINGS nothing is kept, and what the cap removed the cap is
        answerable for (the record's `findings` length against the constant). The
        hygiene counters must not absorb it and read as a hygiene problem."""
        payload = _payload(
            [_a_finding(text=f"f{i}") for i in range(MAX_FINDINGS)]
            + [_a_finding(dimension="vibes"), _a_finding(evidence_chunk_ids=[NEVER_RETRIEVED])]
        )
        out = _review(payload)
        assert len(out.findings) == MAX_FINDINGS
        assert out.malformed_findings_dropped == 0
        assert out.unretrieved_citations_dropped == 0

    def test_an_unparsed_call_counts_no_drops(self) -> None:
        """A reviewer whose JSON never parsed lost everything to `truncated`; the
        hygiene counters must not borrow credit for measuring that."""
        out = _review('{"scor')
        assert out.truncated
        assert out.unretrieved_citations_dropped == 0
        assert out.malformed_findings_dropped == 0

    def test_neither_counter_has_a_default(self) -> None:
        """D18: a field whose absence changes a number's meaning has no default. A
        reviewer output written before these existed cannot load as though the pass
        had been measured and found nothing — which is the same untellable 0 the
        finding is about, arriving through the schema instead of the filter."""
        fields: dict[str, object] = {
            "reviewer": "test-lens",
            "model": "stub",
            "scores": {"soundness": 3, "presentation": 3, "contribution": 3},
            "confidence": 3,
            "findings": [],
        }
        with pytest.raises(ValueError):
            ReviewerOutput.model_validate(fields)
        with pytest.raises(ValueError):
            ReviewerOutput.model_validate({**fields, "unretrieved_citations_dropped": 0})
        with pytest.raises(ValueError):
            ReviewerOutput.model_validate({**fields, "malformed_findings_dropped": 0})
        ReviewerOutput.model_validate(
            {**fields, "unretrieved_citations_dropped": 0, "malformed_findings_dropped": 0}
        )


class TestStructuralQueries:
    MANUSCRIPT = (
        "Intro text here. " * 30
        + "\nMaterials and Methods\nCells were grown in minimal medium and RNA was "
        + "extracted for sequencing analysis with three biological replicates. " * 5
    )

    def test_methods_queries_find_the_methods_section(self) -> None:
        queries = methods_queries(self.MANUSCRIPT, "A title")
        assert queries[0] == "A title"
        assert any("Materials and Methods" in q or "Cells were grown" in q for q in queries)

    def test_novelty_queries_lead_with_title_and_lead(self) -> None:
        queries = novelty_queries(self.MANUSCRIPT, "A title")
        assert queries[0] == "A title"
        assert queries[1].startswith("Intro text here.")

    def test_excerpt_bounded(self) -> None:
        assert len(excerpt("w " * 5000).split()) == EXCERPT_WORDS
