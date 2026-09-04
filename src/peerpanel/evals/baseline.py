"""The single-agent baseline: chain-of-thought plus self-consistency sampling.

The multi-agent literature's central complaint is that this row is always
missing: debate and panel systems "often fail to outperform simple
single-agent baselines such as Chain-of-Thought and Self-Consistency, even
when consuming significantly more inference-time computation"
(arXiv:2502.08788). So the baseline here is that strong shape — one model,
chain-of-thought, sampled repeatedly with findings unioned across samples —
run until it has spent the tokens the panel spent OR hit MAX_SAMPLES, whichever
comes first, through the SAME ledger type. On every committed run the ceiling
came first, so the baseline had LESS compute than the panel; the published
record derives that from the numbers rather than asserting a match.

ONE call of this function is ONE arm: one named local model, run alone. The
planted evaluation calls it once per model in `planted_eval.BASELINE_MODELS`
(`llama3.1:8b`, `qwen2:7b`), each arm given the same budget — the panel's spend
— and each published as its own row under `baseline_system(model)`. That is what
makes "the pass rate for llama3.1:8b" a sentence the record can answer: a single
arm blending the two models would have published a rate belonging to neither,
and the panel row (a two-family mixture by design) already carries that shape.
This function knows nothing about how many arms there are; the models it is
named for are the caller's to choose, and the record names the provider it ran.

What the baseline is NOT given equally: retrieval. It retrieves ONCE, on the
title, and reads EVIDENCE_CHUNKS chunks; the panel's reviewers each retrieve on
three queries and its verifier retrieves per claim. That asymmetry favours the
panel and is enumerated in results/RESULTS.md.

If the panel does not beat this, the table says so.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from peerpanel.agents.json_call import call_json
from peerpanel.agents.reviewer_base import (
    EVIDENCE_CHUNKS,
    EVIDENCE_WORDS,
    REVIEW_MAX_TOKENS,
    excerpt,
)
from peerpanel.agents.schemas import QUOTE_MAX_CHARS, CallStats
from peerpanel.evals.planted import scored_text
from peerpanel.providers.base import ChatProvider, TokenLedger

MAX_SAMPLES = 12

SYSTEM = (
    "You are reviewing a scientific manuscript excerpt for errors and weaknesses. "
    "Think step by step about the methods, the statistics, the internal consistency of "
    "the reported results, and the citations, then list concrete findings. In `quote`, "
    "copy the shortest exact phrase of problematic text each finding is about (at most "
    "25 words). Report only problems the text supports."
)

SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "quote": {"type": "string", "maxLength": QUOTE_MAX_CHARS},
                },
                "required": ["text", "quote"],
            },
        }
    },
    "required": ["findings"],
}


class BaselineResult(BaseModel):
    samples: int
    # Samples whose JSON could not be parsed after the retry, and so contributed nothing.
    # Counted because the panel's equivalent — a reviewer that returns no findings — is
    # counted too: a comparison where one arm can quietly lose a call and the other cannot
    # is not a comparison. Zero is the expected value and worth seeing.
    unparsed_samples: int
    finding_texts: list[str]
    total_tokens: int
    # Samples plus retries; every prompt was read whole iff smallest_headroom_tokens > 0.
    model_calls: CallStats


def _truncate(text: str, words: int = EVIDENCE_WORDS) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + (" …" if len(parts) > words else "")


def run_baseline(
    *,
    manuscript_text: str,
    retrieve: Callable[[str], list[tuple[str, str]]],
    title: str,
    provider: ChatProvider,
    target_tokens: int,
    max_samples: int = MAX_SAMPLES,
) -> BaselineResult:
    """Sample until the token budget matches the panel's or MAX_SAMPLES, unioning findings."""
    ledger = TokenLedger()
    context_lines = [
        f"[{cid}] {_truncate(text)}" for cid, text in retrieve(title)[:EVIDENCE_CHUNKS]
    ]
    user = f"MANUSCRIPT EXCERPT:\n{excerpt(manuscript_text)}\n\nLITERATURE CONTEXT:\n" + (
        "\n".join(context_lines) or "(none)"
    )
    finding_texts: list[str] = []
    samples = 0
    unparsed = 0
    while samples < max_samples and ledger.total_tokens < target_tokens:
        data = call_json(
            provider,
            ledger,
            system=SYSTEM,
            user=user,
            schema=SCHEMA,
            # The same output budget and retry rule as a panel reviewer.
            max_tokens=REVIEW_MAX_TOKENS,
            # Sample 1 is the deterministic core; later samples vary for
            # self-consistency, which is what makes this the strong baseline.
            temperature=0.0 if samples == 0 else 0.7,
        )
        samples += 1
        if data is None:
            unparsed += 1
            continue
        findings = data.get("findings")
        for finding in findings if isinstance(findings, list) else []:
            if isinstance(finding, dict) and finding.get("text"):
                merged = scored_text(str(finding["text"]), str(finding.get("quote") or ""))
                if merged not in finding_texts:
                    finding_texts.append(merged)
    return BaselineResult(
        samples=samples,
        unparsed_samples=unparsed,
        finding_texts=finding_texts,
        total_tokens=ledger.total_tokens,
        model_calls=CallStats.from_ledger(ledger),
    )
