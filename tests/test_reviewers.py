"""Reviewer tests — grounded prompts, hygiene, structural query derivation."""

from __future__ import annotations

import json
import re
from pathlib import Path

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
