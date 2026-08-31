"""Panel schemas: what reviewers, the verifier, the deterministic lens and the
converger produce. Rubric dimensions are the field's own (soundness /
presentation / contribution + confidence); reviewer lenses are structurally
different (disjoint retrieval scope, different model family), never
prompt-persona different.
"""

from __future__ import annotations

from pydantic import BaseModel

RUBRIC_DIMENSIONS = ("soundness", "presentation", "contribution")

VERDICT_SUPPORTS = "SUPPORTS"
VERDICT_REFUTES = "REFUTES"
VERDICT_NEI = "NOT_ENOUGH_INFO"
VERDICTS = (VERDICT_SUPPORTS, VERDICT_REFUTES, VERDICT_NEI)

CONFLICT_TYPES = ("propositional", "sentiment", "evidence")


class ReviewFinding(BaseModel):
    dimension: str  # one of RUBRIC_DIMENSIONS
    severity: str  # "major" | "minor"
    text: str
    evidence_chunk_ids: list[str]  # retrieval grounding; empty = ungrounded (flagged)


class ReviewerOutput(BaseModel):
    reviewer: str  # "methods-statistics" | "prior-work-novelty" | ...
    model: str
    scores: dict[str, int]  # rubric dimension -> 1..5
    confidence: int  # 1..5
    findings: list[ReviewFinding]
    truncated: bool = False


class AtomicClaim(BaseModel):
    claim_id: str
    text: str
    source: str  # "manuscript" | a reviewer name


class EvidenceSpan(BaseModel):
    chunk_id: str
    quote: str


class ClaimVerdict(BaseModel):
    claim_id: str
    claim_text: str
    verdict: str  # one of VERDICTS
    evidence: list[EvidenceSpan]
    swap_consistent: bool  # both evidence orders agreed; False forces NEI (abstain)


class DeterministicFinding(BaseModel):
    check: str  # "gene_symbol_corruption" | "reference_integrity" | ...
    severity: str  # "major" | "minor"
    detail: str
    location: str  # line/section hint


class Conflict(BaseModel):
    type: str  # one of CONFLICT_TYPES
    between: list[str]  # reviewer names
    detail: str


class PanelReview(BaseModel):
    manuscript_doi: str
    corpus_manifest: str
    excluded_docs: list[str]
    reviewer_outputs: list[ReviewerOutput]
    verdicts: list[ClaimVerdict]
    deterministic_findings: list[DeterministicFinding]
    conflicts: list[Conflict]
    summary: str  # converger prose — may only cite reviewer findings + evidence
    swap_consistency_rate: float | None  # None below the reporting N
    n_swap_checked: int
    total_tokens: int
    wall_s: float
