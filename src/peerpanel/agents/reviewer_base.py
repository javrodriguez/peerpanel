"""Shared reviewer machinery: retrieve → grounded prompt → structured review.

Reviewers differ STRUCTURALLY (retrieval scope, rubric focus, model family) —
never by persona (personas measurably do not improve performance and read as
naive; the differentiation must be real). Findings must cite provided chunk
ids; ungrounded citations are dropped in the hygiene pass, not trusted — and
COUNTED there, because a drop nobody counts cannot be told apart from a
reviewer that cited nothing (schemas.ReviewerOutput).

Context budget: the prompt is the manuscript EXCERPT (title + lead, 900 words)
plus a handful of truncated evidence chunks — about 4,100 tokens on qwen2's
tokenizer, so it does NOT fit Ollama's default 4,096 window and the provider
sizes the window per call (providers/context.py). What was reviewed is an
excerpt, and the panel review records that honestly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from peerpanel.providers.base import ChatProvider, TokenLedger

from .json_call import call_json
from .schemas import QUOTE_MAX_CHARS, RUBRIC_DIMENSIONS, ReviewerOutput, ReviewFinding

MAX_FINDINGS = 8
EVIDENCE_CHUNKS = 3
EVIDENCE_WORDS = 120
EXCERPT_WORDS = 900
REVIEW_MAX_TOKENS = 1024  # one retry at double if the answer did not parse (json_call)

REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "object",
            "properties": {dim: {"type": "integer"} for dim in RUBRIC_DIMENSIONS},
            "required": list(RUBRIC_DIMENSIONS),
        },
        "confidence": {"type": "integer"},
        "findings": {
            "type": "array",
            "maxItems": MAX_FINDINGS,
            "items": {
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "enum": list(RUBRIC_DIMENSIONS)},
                    "severity": {"type": "string", "enum": ["major", "minor"]},
                    "text": {"type": "string"},
                    "quote": {"type": "string", "maxLength": QUOTE_MAX_CHARS},
                    "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["dimension", "severity", "text", "quote", "evidence_chunk_ids"],
            },
        },
    },
    "required": ["scores", "confidence", "findings"],
}

SYSTEM_TEMPLATE = (
    "You are reviewing a scientific manuscript excerpt for a structured pre-submission "
    "check. Focus: {focus}. Score each rubric dimension (soundness, presentation, "
    "contribution) from 1 (poor) to 5 (excellent) and give your confidence 1-5. Report "
    "at most {max_findings} findings, each tied to a rubric dimension, grounded in the "
    "provided LITERATURE CONTEXT where relevant — cite context chunk ids in "
    "evidence_chunk_ids (use an empty list only for findings about the excerpt itself). "
    "In `quote`, copy the shortest exact phrase of manuscript text each finding is about "
    "(at most 25 words; empty if none). Judge only what the text supports; never invent "
    "facts or citations."
)
# The `quote` ask is the same channel the single-agent baseline gets (evals/baseline.py),
# so no arm is offered a route to the detection token another lacks. What is SCORED is a
# narrower thing, and this comment claimed the old rule long after it changed: the quote
# field has not been scored since the round-3 review found every credited detection was a
# restatement, and under `assertion-v2` any sentence of `text` that is itself a slice of
# the manuscript is stripped too (evals/planted.own_prose). The quote is collected and
# committed so a reader can see what each finding pointed at — it is evidence, not score.

_SEVERITIES = ("major", "minor")


def _clamp_score(value: object, default: int) -> int:
    """1..5 from whatever the model emitted; a non-integer falls back to `default`.

    Constrained decoding makes the schema binding on the Ollama native wire only —
    every other wire can hand back parseable JSON that violates it, and a reviewer
    crashing on a bad score would take the whole panel with it.
    """
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return default
    try:
        number = int(value)
    except ValueError:
        return default
    return max(1, min(5, number))


def _finding(raw: object, seen: set[str]) -> tuple[ReviewFinding | None, int]:
    """One finding and the number of citations hygiene removed from it.

    The finding is None when it is not an object, lacks a rubric dimension or carries
    no text; the caller counts those. The int is separate because a kept finding hides
    its own losses completely — the citations that named a chunk this reviewer was
    never given are simply gone from the list a reader sees, so a finding that cited
    three unretrieved ids and one that cited nothing are the same finding in the
    record. Both counts land on ReviewerOutput.
    """
    if not isinstance(raw, dict):
        return None, 0
    dimension = raw.get("dimension")
    text = raw.get("text")
    if dimension not in RUBRIC_DIMENSIONS or not isinstance(text, str) or not text.strip():
        return None, 0
    severity = raw.get("severity")
    cited = raw.get("evidence_chunk_ids")
    if isinstance(cited, list):
        grounded = [cid for cid in cited if cid in seen]
        dropped = len(cited) - len(grounded)
    else:
        # An absent citation field means nothing was cited, so nothing was removed. A
        # PRESENT one that is not a list is discarded whole (constrained decoding binds
        # on one wire only) and counts as one removal: how many ids it meant is
        # unknowable, and a 0 here would be the silent drop this counter exists to end.
        grounded = []
        dropped = 0 if cited is None else 1
    return (
        ReviewFinding(
            dimension=dimension,
            severity=severity if severity in _SEVERITIES else "minor",
            text=text.strip(),
            quote=str(raw.get("quote") or "").strip(),
            evidence_chunk_ids=grounded,
        ),
        dropped,
    )


def excerpt(manuscript_text: str, words: int = EXCERPT_WORDS) -> str:
    return " ".join(manuscript_text.split()[:words])


def _truncate(text: str, words: int = EVIDENCE_WORDS) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + (" …" if len(parts) > words else "")


def run_reviewer(
    *,
    reviewer_name: str,
    provider: ChatProvider,
    manuscript_text: str,
    retrieve: Callable[[str], list[tuple[str, str]]],
    focus: str,
    queries: list[str],
    ledger: TokenLedger | None = None,
) -> ReviewerOutput:
    """One structured, grounded review. Deterministic hygiene over LLM output."""
    seen: set[str] = set()
    context_lines: list[str] = []
    for query in queries:
        for chunk_id, text in retrieve(query)[:EVIDENCE_CHUNKS]:
            if chunk_id not in seen:
                seen.add(chunk_id)
                context_lines.append(f"[{chunk_id}] {_truncate(text)}")
    context = "\n".join(context_lines) or "(no literature context retrieved)"
    data = call_json(
        provider,
        ledger,
        system=SYSTEM_TEMPLATE.format(focus=focus, max_findings=MAX_FINDINGS),
        user=(
            f"MANUSCRIPT EXCERPT:\n{excerpt(manuscript_text)}\n\n"
            f"LITERATURE CONTEXT (chunks from the corpus):\n{context}"
        ),
        schema=REVIEW_SCHEMA,
        max_tokens=REVIEW_MAX_TOKENS,
    )
    if data is None:
        return ReviewerOutput(
            reviewer=reviewer_name,
            model=provider.name,
            scores={},
            confidence=0,
            findings=[],
            # Nothing parsed, so hygiene removed nothing: the loss this record carries
            # is `truncated`, and these two must not borrow credit for measuring it.
            unretrieved_citations_dropped=0,
            malformed_findings_dropped=0,
            truncated=True,
        )
    raw_scores = data.get("scores")
    raw_scores = raw_scores if isinstance(raw_scores, dict) else {}
    scores = {dim: _clamp_score(raw_scores.get(dim), 1) for dim in RUBRIC_DIMENSIONS}
    raw_findings = data.get("findings")
    raw_findings = raw_findings if isinstance(raw_findings, list) else []
    findings: list[ReviewFinding] = []
    citations_dropped = 0
    findings_dropped = 0
    for raw in raw_findings:
        # MAX_FINDINGS is a hard cap in code, not a hope pinned on the schema's maxItems.
        # Both counters describe the hygiene applied to the findings this record CARRIES:
        # past the cap nothing is kept, and what the cap removed the cap is answerable
        # for — the record's own `findings` length against this constant shows it.
        if len(findings) >= MAX_FINDINGS:
            break
        finding, dropped = _finding(raw, seen)
        if finding is None:
            findings_dropped += 1
            continue
        findings.append(finding)
        citations_dropped += dropped
    return ReviewerOutput(
        reviewer=reviewer_name,
        model=provider.name,
        scores=scores,
        confidence=_clamp_score(data.get("confidence"), 1),
        findings=findings,
        unretrieved_citations_dropped=citations_dropped,
        malformed_findings_dropped=findings_dropped,
    )
