"""Panel schemas: what reviewers, the verifier, the deterministic lens and the
converger produce. Rubric dimensions are the field's own (soundness /
presentation / contribution + confidence); reviewer lenses are structurally
different (disjoint retrieval scope, different model family), never
prompt-persona different.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

RUBRIC_DIMENSIONS = ("soundness", "presentation", "contribution")

VERDICT_SUPPORTS = "SUPPORTS"
VERDICT_REFUTES = "REFUTES"
VERDICT_NEI = "NOT_ENOUGH_INFO"
VERDICTS = (VERDICT_SUPPORTS, VERDICT_REFUTES, VERDICT_NEI)

CONFLICT_TYPES = ("propositional", "sentiment", "evidence")

# The longest `quote` a finding may carry, enforced in the JSON schema (maxLength) so
# constrained decoding cannot copy a whole paragraph into every finding and run the
# output budget dry — measured on qwen2:7b, which did exactly that once the reviewer
# prompt was read whole (DECISIONS.md D19). 240 characters is a sentence: enough to
# name the planted token the evaluation scores on, both arms alike.
QUOTE_MAX_CHARS = 240


class ReviewFinding(BaseModel):
    dimension: str  # one of RUBRIC_DIMENSIONS
    severity: str  # "major" | "minor"
    text: str
    quote: str = ""  # a short exact phrase of manuscript text the finding is about ("" if none)
    evidence_chunk_ids: list[str]  # retrieval grounding; empty = ungrounded (flagged)


class ReviewerOutput(BaseModel):
    reviewer: str  # "methods-statistics" | "prior-work-novelty" | ...
    model: str
    scores: dict[str, int]  # rubric dimension -> 1..5
    confidence: int  # 1..5
    findings: list[ReviewFinding]
    # Citation ids the hygiene pass removed from the findings above: ids naming a chunk
    # this reviewer was never given. Recorded because the removal leaves no trace in the
    # finding it was removed from — RESULTS.md reads the surviving citations as reviewer
    # behaviour ("of 16 findings in a run, 1 cites a retrieved chunk at all"), and
    # without this a reviewer that cited nothing cannot be told apart from one whose
    # citations this pass deleted. Required, no default: a 0 written by a default would
    # be that same untellable number wearing a measurement's clothes.
    unretrieved_citations_dropped: int
    # Whole findings the hygiene pass discarded before the list above: not an object, no
    # rubric dimension, or no text. Recorded for the reason `unparsed_calls` is
    # (evals/planted_eval.py) — it silently lowers the count of findings a run is judged
    # on, so the denominator of every per-finding number here is partly this filter's
    # doing. Required for the same reason as the field above.
    malformed_findings_dropped: int
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
    swapped: bool  # was this claim ACTUALLY judged twice? A claim with no evidence
    # has no second ordering, so it is trivially 'consistent' and must not pad the
    # rate's denominator with agreement it was never at risk of losing. Required, no
    # default: a record that predates the field cannot load as if it had been judged.


class DeterministicFinding(BaseModel):
    check: str  # "gene_symbol_corruption" | "reference_integrity" | ...
    severity: str  # "major" | "minor"
    detail: str
    location: str  # line/section hint


class Conflict(BaseModel):
    type: str  # one of CONFLICT_TYPES
    between: list[str]  # reviewer names
    detail: str


class SwapStat(BaseModel):
    """Swap-consistency over one population of claims: the rate and its n."""

    rate: float | None  # None below the reporting N
    n: int


class CallStats(BaseModel):
    """What one ledger saw: proof, per record, that no prompt was cut.

    `smallest_headroom_tokens` is the proof — the margin between a prompt and the
    window that prompt ran in, smallest across the record's calls; positive means
    every call was read whole. `window_sources` says what that margin was measured
    AGAINST: the distinct WINDOW_SOURCE_* strings the calls named, each naming the
    field one wire set or read for the call it served (providers/base.py). The native
    wire sets `options.num_ctx` per call; the OpenAI-compatible wire reads the loaded
    runner's `context_length` before every call, because a native call in between can
    change it. A margin whose source is unnamed cannot be checked by a reader, so the
    ledger refuses a window without one and this list is EMPTY only when `calls` is 0.
    The other three describe the run's shape and cannot prove it on their own, because
    the largest prompt and the smallest window are usually different calls. Required
    on every record — one that predates the fields cannot load as if it had been
    measured (DECISIONS.md D19 withdrew every such record).
    """

    calls: int
    largest_prompt_tokens: int
    smallest_context: int | None  # None only from a wire that does not report its window
    smallest_headroom_tokens: int | None  # window minus prompt, per call, smallest
    window_sources: list[str]  # sorted, distinct; [] only when calls == 0

    @classmethod
    def from_ledger(cls, ledger: Any) -> CallStats:
        return cls(
            calls=ledger.calls,
            largest_prompt_tokens=ledger.largest_prompt_tokens,
            smallest_context=ledger.smallest_context,
            smallest_headroom_tokens=ledger.smallest_headroom_tokens,
            window_sources=ledger.window_source_list,
        )


class PanelReview(BaseModel):
    manuscript_doi: str
    corpus_manifest: str
    excluded_docs: list[str]
    dropped_chunks: int = 0  # what THIS run's index actually removed
    reviewer_outputs: list[ReviewerOutput]
    verdicts: list[ClaimVerdict]
    deterministic_findings: list[DeterministicFinding]
    conflicts: list[Conflict]
    summary: str  # converger prose — may only cite reviewer findings + evidence
    swap_consistency_rate: float | None  # None below the reporting N
    n_swap_checked: int  # claims that were judged twice (the rate's denominator)
    swap_by_source: dict[str, SwapStat] = {}  # "manuscript" | reviewer name -> its rate
    total_tokens: int
    wall_s: float
    model_calls: CallStats  # every prompt read whole iff smallest_headroom_tokens > 0
    # Role -> provider name for every model that served this review. The reviewers were
    # already named per output; the verifier and converger were not named anywhere a
    # reader could check, which is a run condition the record must carry.
    models: dict[str, str]
