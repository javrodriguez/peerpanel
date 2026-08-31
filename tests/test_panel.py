"""Orchestrator tests — conflict detection and convergence, all deterministic."""

from __future__ import annotations

from peerpanel.agents.schemas import (
    ClaimVerdict,
    EvidenceSpan,
    ReviewerOutput,
    ReviewFinding,
)
from peerpanel.orchestration import detect_conflicts
from peerpanel.orchestration.panel import converge_summary
from peerpanel.providers.base import ChatResponse, TokenLedger


def _output(reviewer: str, soundness: int, findings: list[str] | None = None) -> ReviewerOutput:
    return ReviewerOutput(
        reviewer=reviewer,
        model="stub",
        scores={"soundness": soundness, "presentation": 3, "contribution": 3},
        confidence=3,
        findings=[
            ReviewFinding(dimension="soundness", severity="major", text=t, evidence_chunk_ids=[])
            for t in (findings or [])
        ],
    )


def _verdict(claim_id: str, verdict: str, text: str = "claim") -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=claim_id, claim_text=text, verdict=verdict,
        evidence=[EvidenceSpan(chunk_id="d:0:x", quote="q")], swap_consistent=True,
    )


class TestConflicts:
    def test_score_gap_is_a_sentiment_conflict(self) -> None:
        conflicts = detect_conflicts(
            [_output("methods-statistics", 5), _output("prior-work-novelty", 2)], []
        )
        assert [c.type for c in conflicts] == ["sentiment"]
        assert "soundness" in conflicts[0].detail

    def test_close_scores_no_conflict(self) -> None:
        assert detect_conflicts([_output("a", 4), _output("b", 3)], []) == []

    def test_refuted_reviewer_claim_is_an_evidence_conflict(self) -> None:
        verdicts = [
            _verdict("manuscript:0:abc", "REFUTES"),  # manuscript refutation: not a conflict
            _verdict("prior-work-novelty:1:panel000", "REFUTES", "overlaps X"),
        ]
        conflicts = detect_conflicts([], verdicts)
        assert len(conflicts) == 1
        assert conflicts[0].type == "evidence"
        assert "prior-work-novelty" in conflicts[0].between

    def test_empty_inputs(self) -> None:
        assert detect_conflicts([], []) == []


class _StubChat:
    name = "stub-converger"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(self, *, system: str, user: str, json_schema: object = None,
             temperature: float = 0.0, max_tokens: int = 2048) -> ChatResponse:
        self.calls.append(user)
        return ChatResponse(text="Synthesis.", model="stub", prompt_tokens=5, completion_tokens=2)


class TestConverger:
    def test_prompt_carries_only_the_structured_record(self) -> None:
        stub = _StubChat()
        ledger = TokenLedger()
        outputs = [_output("methods-statistics", 4, ["No replication stated."])]
        verdicts = [_verdict("manuscript:0:abc", "SUPPORTS", "growth was measured")]
        summary = converge_summary(outputs, verdicts, [], stub, ledger)  # type: ignore[arg-type]
        assert summary == "Synthesis."
        prompt = stub.calls[0]
        assert "No replication stated." in prompt
        assert "SUPPORTS" in prompt
        assert ledger.calls == 1

    def test_abstained_verdicts_labelled(self) -> None:
        stub = _StubChat()
        verdict = ClaimVerdict(
            claim_id="manuscript:0:x", claim_text="c", verdict="NOT_ENOUGH_INFO",
            evidence=[], swap_consistent=False,
        )
        converge_summary([], [verdict], [], stub, TokenLedger())  # type: ignore[arg-type]
        assert "order-inconsistent -> abstained" in stub.calls[0]
