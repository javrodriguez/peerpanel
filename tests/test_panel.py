"""Orchestrator tests — conflict detection and convergence, all deterministic."""

from __future__ import annotations

import json

from peerpanel.agents.claims_verifier import REVIEWER_CLAIM_RULE, claim_source
from peerpanel.agents.schemas import (
    ClaimVerdict,
    EvidenceSpan,
    ReviewerOutput,
    ReviewFinding,
)
from peerpanel.orchestration import detect_conflicts
from peerpanel.orchestration.panel import converge_summary, reviewer_claims
from peerpanel.providers.base import ChatResponse, TokenLedger


def _output(reviewer: str, soundness: int, findings: list[str] | None = None) -> ReviewerOutput:
    return ReviewerOutput(
        reviewer=reviewer,
        model="stub",
        scores={"soundness": soundness, "presentation": 3, "contribution": 3},
        confidence=3,
        # This stub builds clean output — nothing was filtered on the way in — so both
        # hygiene counters are 0. They are required rather than defaulted (D18), which is
        # why a fixture has to state them: a silent 0 is the untellable number again.
        unretrieved_citations_dropped=0,
        malformed_findings_dropped=0,
        findings=[
            ReviewFinding(dimension="soundness", severity="major", text=t, evidence_chunk_ids=[])
            for t in (findings or [])
        ],
    )


def _verdict(
    claim_id: str, verdict: str, text: str = "claim", grounded: bool = True
) -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=claim_id,
        claim_text=text,
        verdict=verdict,
        evidence=[EvidenceSpan(chunk_id="d:0:x", quote="q")] if grounded else [],
        swap_consistent=True,
        swapped=True,
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

    def test_ungrounded_refutes_is_not_an_evidence_conflict(self) -> None:
        """A REFUTES with no grounded span is the judge's word alone; presenting
        it as a documented contradiction would launder an ungrounded verdict."""
        verdicts = [
            _verdict("prior-work-novelty:1:panel000", "REFUTES", "overlaps X", grounded=False)
        ]
        assert detect_conflicts([], verdicts) == []

    def test_empty_inputs(self) -> None:
        assert detect_conflicts([], []) == []


class _StubChat:
    name = "stub-converger"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(
        self,
        *,
        system: str,
        user: str,
        json_schema: object = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse:
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
            claim_id="manuscript:0:x",
            claim_text="c",
            verdict="NOT_ENOUGH_INFO",
            evidence=[],
            swap_consistent=False,
            swapped=True,
        )
        converge_summary([], [verdict], [], stub, TokenLedger())  # type: ignore[arg-type]
        assert "order-inconsistent -> abstained" in stub.calls[0]


class _ClaimStub:
    """Answers decomposition with one claim per line of the passage it is given."""

    name = "stub-decomposer"

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def chat(
        self,
        *,
        system: str,
        user: str,
        json_schema: object = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        self.calls.append({"system": system, "user": user})
        passage = user.split("PASSAGE:\n", 1)[1]
        claims = [line for line in passage.splitlines() if line.strip()]
        return ChatResponse(
            text=json.dumps({"claims": claims}), model="stub", prompt_tokens=5, completion_tokens=2
        )


def _mixed_output(reviewer: str) -> ReviewerOutput:
    return ReviewerOutput(
        reviewer=reviewer,
        model="stub",
        scores={"soundness": 3, "presentation": 3, "contribution": 3},
        confidence=3,
        unretrieved_citations_dropped=0,
        malformed_findings_dropped=0,
        findings=[
            ReviewFinding(
                dimension="soundness",
                severity="major",
                text="Dcr1 is dispensable for silencing.",
                evidence_chunk_ids=[],
            ),
            ReviewFinding(
                dimension="presentation",
                severity="minor",
                text="The introduction is well organised.",
                evidence_chunk_ids=[],
            ),
            ReviewFinding(
                dimension="contribution",
                severity="minor",
                text="Prior work already showed this in S. pombe.",
                evidence_chunk_ids=[],
            ),
        ],
    )


class TestReviewerClaims:
    """A reviewer's opinion is not a claim; the propositions inside it are."""

    def test_only_soundness_and_contribution_findings_are_candidates(self) -> None:
        stub = _ClaimStub()
        claims = reviewer_claims([_mixed_output("methods-statistics")], stub)  # type: ignore[arg-type]
        assert [c.text for c in claims] == [
            "Dcr1 is dispensable for silencing.",
            "Prior work already showed this in S. pombe.",
        ]
        assert "well organised" not in stub.calls[0]["user"]

    def test_claims_carry_the_reviewer_as_source_in_the_id(self) -> None:
        claims = reviewer_claims([_mixed_output("prior-work-novelty")], _ClaimStub())  # type: ignore[arg-type]
        assert {c.source for c in claims} == {"prior-work-novelty"}
        assert all(c.claim_id.startswith("prior-work-novelty:") for c in claims)
        assert claim_source(claims[0].claim_id) == "prior-work-novelty"

    def test_findings_go_through_the_claim_splitter_under_the_reviewer_rule(self) -> None:
        stub = _ClaimStub()
        reviewer_claims([_mixed_output("methods-statistics")], stub)  # type: ignore[arg-type]
        assert len(stub.calls) == 1  # one decomposition per reviewer
        assert REVIEWER_CLAIM_RULE.strip() in stub.calls[0]["system"]

    def test_a_reviewer_with_no_candidate_findings_costs_no_call(self) -> None:
        stub = _ClaimStub()
        assert reviewer_claims([_output("methods-statistics", 3)], stub) == []  # type: ignore[arg-type]
        assert stub.calls == []

    def test_reviewer_findings_never_become_claims_verbatim(self) -> None:
        """The pre-fix shape appended every finding as a claim with a fixed
        `panel000` hash; the splitter's ids carry a real passage hash."""
        claims = reviewer_claims([_mixed_output("methods-statistics")], _ClaimStub())  # type: ignore[arg-type]
        assert all(not c.claim_id.endswith(":panel000") for c in claims)
