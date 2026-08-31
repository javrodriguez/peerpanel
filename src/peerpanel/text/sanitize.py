"""The injection screen — every document (corpus AND manuscripts) passes
through here before chunking.

Scientific text is untrusted input to the reviewer agents: hidden prompts in
manuscripts measurably flip LLM reviews (arXiv:2509.10248). The screen is
deterministic: control/zero-width characters are removed, instruction-shaped
lines are flagged and neutralised (prefixed, never silently dropped), and
every action is returned as a finding so runs can report what was screened.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Zero-width and bidi-control characters that can hide text from a human eye.
_INVISIBLES = re.compile(r"[​‌‍⁠﻿‪-‮⁦-⁩]")

# Instruction-shaped patterns that have no place in scientific prose.
_INJECTION_PATTERNS = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"\bas\s+an?\s+ai\b.{0,40}\b(accept|approve|rate|score)", re.IGNORECASE),
    re.compile(r"(accept|approve)\s+this\s+(paper|manuscript|submission)", re.IGNORECASE),
)

NEUTRALISED_PREFIX = "[screened: instruction-shaped line quoted as data] "


@dataclass(frozen=True)
class Finding:
    kind: str  # "invisible_chars" | "control_chars" | "instruction_line"
    line_no: int
    detail: str


def sanitize(text: str) -> tuple[str, list[Finding]]:
    """Return (clean_text, findings). Deterministic; never drops content lines."""
    findings: list[Finding] = []
    out_lines: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if _INVISIBLES.search(line):
            findings.append(Finding("invisible_chars", line_no, "zero-width/bidi removed"))
            line = _INVISIBLES.sub("", line)
        cleaned = "".join(
            ch for ch in line if ch == "\t" or not unicodedata.category(ch).startswith("C")
        )
        if cleaned != line:
            findings.append(Finding("control_chars", line_no, "control characters removed"))
            line = cleaned
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(line):
                findings.append(Finding("instruction_line", line_no, pattern.pattern))
                line = NEUTRALISED_PREFIX + line
                break
        out_lines.append(line)
    return "\n".join(out_lines), findings
