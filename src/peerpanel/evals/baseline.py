"""The equal-compute single-agent baseline.

The multi-agent literature's central complaint is that this row is always
missing: debate and panel systems "often fail to outperform simple
single-agent baselines such as Chain-of-Thought and Self-Consistency, even
when consuming significantly more inference-time computation"
(arXiv:2502.08788). So the baseline here is the strong version — one model,
the same retrieval budget, chain-of-thought plus self-consistency sampling —
run until it has spent approximately the SAME tokens the panel spent, through
the SAME ledger type. Findings are unioned across samples (self-consistency).

If the panel does not beat this, the table says so.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import BaseModel

from peerpanel.agents.reviewer_base import EVIDENCE_CHUNKS, EVIDENCE_WORDS, excerpt
from peerpanel.providers.base import ChatProvider, TokenLedger

MAX_SAMPLES = 12

SYSTEM = (
    "You are reviewing a scientific manuscript excerpt for errors and weaknesses. "
    "Think step by step about the methods, the statistics, the internal consistency of "
    "the reported results, and the citations, then list concrete findings. Quote the "
    "exact problematic text in each finding. Report only problems the text supports."
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
                    "quote": {"type": "string"},
                },
                "required": ["text", "quote"],
            },
        }
    },
    "required": ["findings"],
}


class BaselineResult(BaseModel):
    samples: int
    finding_texts: list[str]
    total_tokens: int


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
    """Sample until the token budget matches the panel's, unioning findings."""
    ledger = TokenLedger()
    context_lines = [
        f"[{cid}] {_truncate(text)}" for cid, text in retrieve(title)[:EVIDENCE_CHUNKS]
    ]
    user = (
        f"MANUSCRIPT EXCERPT:\n{excerpt(manuscript_text)}\n\n"
        f"LITERATURE CONTEXT:\n" + ("\n".join(context_lines) or "(none)")
    )
    finding_texts: list[str] = []
    samples = 0
    while samples < max_samples and ledger.total_tokens < target_tokens:
        response = provider.chat(
            system=SYSTEM,
            user=user,
            json_schema=SCHEMA,
            # Sample 1 is the deterministic core; later samples vary for
            # self-consistency, which is what makes this the strong baseline.
            temperature=0.0 if samples == 0 else 0.7,
            max_tokens=1024,
        )
        ledger.record(response)
        samples += 1
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            continue
        for finding in data.get("findings", []):
            if isinstance(finding, dict) and finding.get("text"):
                merged = f"{finding['text']} {finding.get('quote', '')}".strip()
                if merged not in finding_texts:
                    finding_texts.append(merged)
    return BaselineResult(
        samples=samples, finding_texts=finding_texts, total_tokens=ledger.total_tokens
    )
