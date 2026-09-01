"""Planted-error generation: seeded, documented, and held out by construction.

Errors are planted into a manuscript that is NOT represented in the retrieval
corpus (the deferred caprin manuscript, whose twin is deliberately not a demo
corpus member — DECISIONS D8). That is load-bearing: planting errors in a
paper whose unperturbed original is retrievable measures near-duplicate
diffing, not scientific error detection, and would make the whole evaluation
gameable.

Every perturbation records the exact original span, the exact replacement, and
a detection token — the string a reviewer would have to name to be credited
with the catch. Nothing about the matching rule is fuzzy; its limits are
stated in `detect`.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

from pydantic import BaseModel

SEED = 20260831


class PlantedError(BaseModel):
    error_id: str
    kind: str
    original: str
    replacement: str
    detection_token: str  # what a finding must name to count as a catch
    line_no: int


class PlantedManuscript(BaseModel):
    source_file: str
    text: str
    errors: list[PlantedError]
    skipped_kinds: dict[str, str] = {}  # kind -> why nothing could be planted


_GENE = re.compile(r"\b([A-Z][a-z]{2}[0-9]{1,2}|[A-Z]{2,6}[0-9]{1,2})\b")
_PVALUE = re.compile(r"\bp\s*[<=]\s*0\.0*[0-9]+\b", re.IGNORECASE)
_PERCENT = re.compile(r"\b([0-9]{1,2}(?:\.[0-9])?)%")
_DIRECTION = re.compile(
    r"\b(increased|decreased|higher|lower|upregulated|downregulated)\b", re.IGNORECASE
)
_FLIP = {
    "increased": "decreased", "decreased": "increased",
    "higher": "lower", "lower": "higher",
    "upregulated": "downregulated", "downregulated": "upregulated",
}


def _lines(text: str) -> list[str]:
    return text.splitlines()


def plant_errors(
    text: str, source_file: str, n_per_kind: int = 1, window_words: int | None = None
) -> PlantedManuscript:
    """Plant a seeded, documented set of errors. Deterministic for a given input.

    `window_words` confines planting to the first N words — the SAME window the
    reviewers read. Without it, errors land in text no arm ever sees and the
    benchmark reports a score it structurally cannot earn: measured on the
    shipped subject, planted sites fell at words 1196-6636 against a 900-word
    reviewer excerpt, so not one error was visible to either arm.
    """
    rng = random.Random(SEED)
    lines = _lines(text)
    if window_words is not None:
        cumulative = 0
        last = 0
        for idx, line in enumerate(lines):
            cumulative += len(line.split())
            if cumulative > window_words:
                break
            last = idx
        plantable = last + 1
    else:
        plantable = len(lines)
    errors: list[PlantedError] = []

    def _apply(line_no: int, original: str, replacement: str, kind: str, token: str) -> None:
        lines[line_no] = lines[line_no].replace(original, replacement, 1)
        errors.append(
            PlantedError(
                error_id=f"{kind}-{len(errors)}",
                kind=kind,
                original=original,
                replacement=replacement,
                detection_token=token,
                line_no=line_no + 1,
            )
        )

    # 1. Gene-symbol swap: a real symbol replaced by a plausible neighbour.
    candidates = [
        (i, m.group(1))
        for i, line in enumerate(lines[:plantable])
        for m in [_GENE.search(line)]
        if m and len(line.split()) > 12
    ]
    for i, symbol in rng.sample(candidates, min(n_per_kind, len(candidates))):
        digits = re.search(r"[0-9]+$", symbol)
        wrong = (
            symbol[: digits.start()] + str(int(digits.group()) + 1) if digits else symbol + "2"
        )
        _apply(i, symbol, wrong, "gene_symbol_swap", wrong)

    # 2. Effect-direction flip: the classic reviewer catch.
    directions = [
        (i, m.group(1))
        for i, line in enumerate(lines[:plantable])
        for m in [_DIRECTION.search(line)]
        if m
    ]
    for i, word in rng.sample(directions, min(n_per_kind, len(directions))):
        _apply(i, word, _FLIP[word.lower()], "effect_direction_flip", _FLIP[word.lower()])

    # 3. Statistical impossibility: a p-value above 1.
    pvalues = [
        (i, m.group(0))
        for i, line in enumerate(lines[:plantable])
        for m in [_PVALUE.search(line)]
        if m
    ]
    for i, pval in rng.sample(pvalues, min(n_per_kind, len(pvalues))):
        _apply(i, pval, "p = 1.34", "impossible_pvalue", "p = 1.34")

    # 4. Fabricated citation: a plausible-looking reference to nothing.
    body = [i for i, line in enumerate(lines[:plantable]) if len(line.split()) > 25]
    for i in rng.sample(body, min(n_per_kind, len(body))):
        fake = " (Hollingsworth and Vance, 2019, Nat. Metab. 7:e91188)"
        lines[i] = lines[i].rstrip() + fake
        errors.append(
            PlantedError(
                error_id=f"fabricated_citation-{len(errors)}",
                kind="fabricated_citation",
                original="",
                replacement=fake.strip(),
                detection_token="Hollingsworth",
                line_no=i + 1,
            )
        )

    # 5. Unit/magnitude error: a percentage pushed out of range.
    percents = [
        (i, m.group(0))
        for i, line in enumerate(lines[:plantable])
        for m in [_PERCENT.search(line)]
        if m
    ]
    for i, pct in rng.sample(percents, min(n_per_kind, len(percents))):
        _apply(i, pct, "412%", "impossible_percentage", "412%")

    planted_kinds = {e.kind for e in errors}
    skipped = {
        kind: "no candidate site inside the reviewed window"
        for kind in (
            "gene_symbol_swap", "effect_direction_flip", "impossible_pvalue",
            "fabricated_citation", "impossible_percentage",
        )
        if kind not in planted_kinds
    }
    return PlantedManuscript(
        source_file=source_file,
        text="\n".join(lines),
        errors=errors,
        skipped_kinds=skipped,
    )


def detect(error: PlantedError, finding_texts: list[str]) -> bool:
    """Was this planted error caught?

    The rule is deliberately literal: some finding must NAME the planted token
    (case-insensitive substring). That under-credits a reviewer who describes
    the problem without quoting it ("the reported significance is impossible"
    does not count for `p = 1.34`), and it cannot credit a reviewer for
    catching something it never wrote down. Both directions are stated so the
    reported recall is read as a floor, not a ceiling.
    """
    token = error.detection_token.lower()
    return any(token in text.lower() for text in finding_texts)


def write_planted(path: Path, planted: PlantedManuscript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(planted.model_dump_json(indent=1) + "\n")
