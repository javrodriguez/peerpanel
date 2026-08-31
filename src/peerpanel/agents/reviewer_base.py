"""Shared reviewer machinery: retrieve → grounded prompt → structured review.

Reviewers differ STRUCTURALLY (retrieval scope, rubric focus, model family) —
never by persona (personas measurably do not improve performance and read as
naive; the differentiation must be real). Findings must cite provided chunk
ids; ungrounded citations are dropped in the hygiene pass, not trusted.

Context budget: everything is sized to the 4096-token serving slot — the
manuscript EXCERPT (title + lead) plus a handful of truncated evidence chunks.
What was reviewed is therefore an excerpt, and the panel review records that
honestly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from peerpanel.providers.base import ChatProvider, TokenLedger

from .schemas import RUBRIC_DIMENSIONS, ReviewerOutput, ReviewFinding

MAX_FINDINGS = 8
EVIDENCE_CHUNKS = 3
EVIDENCE_WORDS = 120
EXCERPT_WORDS = 900

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
                    "evidence_chunk_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["dimension", "severity", "text", "evidence_chunk_ids"],
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
    "Judge only what the text supports; never invent facts or citations."
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
    response = provider.chat(
        system=SYSTEM_TEMPLATE.format(focus=focus, max_findings=MAX_FINDINGS),
        user=(
            f"MANUSCRIPT EXCERPT:\n{excerpt(manuscript_text)}\n\n"
            f"LITERATURE CONTEXT (chunks from the corpus):\n{context}"
        ),
        json_schema=REVIEW_SCHEMA,
        temperature=0.0,
        max_tokens=1024,
    )
    if ledger is not None:
        ledger.record(response)
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError:
        return ReviewerOutput(
            reviewer=reviewer_name, model=provider.name, scores={}, confidence=0,
            findings=[], truncated=True,
        )
    scores = {
        dim: max(1, min(5, int(data.get("scores", {}).get(dim, 0)) or 1))
        for dim in RUBRIC_DIMENSIONS
    }
    findings = [
        ReviewFinding(
            dimension=f["dimension"],
            severity=f["severity"],
            text=f["text"].strip(),
            evidence_chunk_ids=[cid for cid in f.get("evidence_chunk_ids", []) if cid in seen],
        )
        for f in data.get("findings", [])
        if isinstance(f, dict) and f.get("dimension") in RUBRIC_DIMENSIONS and f.get("text")
    ]
    return ReviewerOutput(
        reviewer=reviewer_name,
        model=provider.name,
        scores=scores,
        confidence=max(1, min(5, int(data.get("confidence", 1)))),
        findings=findings,
    )
