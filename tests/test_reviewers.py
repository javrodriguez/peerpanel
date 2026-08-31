"""Reviewer tests — grounded prompts, hygiene, structural query derivation."""

from __future__ import annotations

from peerpanel.agents.methods_reviewer import methods_queries
from peerpanel.agents.novelty_reviewer import novelty_queries
from peerpanel.agents.reviewer_base import EXCERPT_WORDS, REVIEW_SCHEMA, excerpt, run_reviewer
from peerpanel.providers.base import ChatResponse, TokenLedger


class _StubChat:
    name = "stub-chat"

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def chat(self, *, system: str, user: str, json_schema: object = None,
             temperature: float = 0.0, max_tokens: int = 2048) -> ChatResponse:
        self.calls.append({"system": system, "user": user, "schema": json_schema})
        return ChatResponse(text=self.payload, model="stub", prompt_tokens=7, completion_tokens=3)


GOOD = (
    '{"scores": {"soundness": 4, "presentation": 3, "contribution": 4}, "confidence": 3,'
    ' "findings": [{"dimension": "soundness", "severity": "major",'
    ' "text": "No replication stated.", "evidence_chunk_ids": ["docA:0:aaaa0000"]},'
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
            reviewer_name="test-lens", provider=stub,  # type: ignore[arg-type]
            manuscript_text="word " * 2000, retrieve=_retrieve,
            focus="testing", queries=["q1"], ledger=ledger,
        )
        return out, stub, ledger

    def test_prompt_carries_excerpt_context_and_schema_no_persona(self) -> None:
        _, stub, _ = self._run()
        call = stub.calls[0]
        assert "[docA:0:aaaa0000]" in str(call["user"])
        assert call["schema"] == REVIEW_SCHEMA
        assert "persona" not in str(call["system"]).lower()
        assert len(str(call["user"]).split()) < 1400  # slot-context budget respected

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
