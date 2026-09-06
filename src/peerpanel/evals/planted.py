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

from peerpanel.text.chunks import sentences

SEED = 20260831
DETECTION_RULE = "assertion-v2"
# v1 -> v2 because WHAT is scored changed, not how it is judged. Under v1 a finding's
# whole `text` reached `asserts` and `detect`; under v2 only `own_prose(text, manuscript)`
# does — the sentences of it that are not verbatim manuscript. `ASSERTION_CUES`, the
# negation scope and the same-sentence clause are byte-identical to v1. The id is bumped
# so no record can carry a v1 rule name over a v2 count.
#
# It stays "assertion-v2" through round 5's terminal-punctuation repair (`own_prose`,
# below). That change corrects WHAT the v2 rule was already trying to remove — quotation
# — rather than redefining what is scored or how it is judged: the rule is still "the
# `own_prose` residue reaches `asserts` and `detect`", the cue list, the negation scope
# and the same-sentence clause are all untouched, and the repair can only ever remove
# more text before those rules run. A record carrying "assertion-v2" therefore still
# describes the rule that produced its count; only the count moves, downwards, and the
# regeneration that follows the repair is what makes every record agree with it again.

# The assertion rule, frozen before any control was run against it (plan D-1) and
# implemented exactly as specified: one word-anchored alternation, in this order,
# joined into a single pattern. Order is load-bearing — the multi-word cues that
# CARRY a negation ("not correct", "does not exist", "cannot verify") must match as
# whole phrases, so that the negation sits INSIDE the cue rather than in the window
# scanned before it. Domain collisions were removed on purpose: `invert` (inverted
# microscope), `opposite` (an opposite effect), bare `revers` (reverse transcribed),
# `no such` (no such enrichment was observed), `suspicious` and `questionable` all
# occur in these manuscripts as ordinary vocabulary, and `error` keeps its `error
# bars` exception for the same reason. Never tune this list to make a control pass:
# a control that trips is a finding about the rule (requirement 7d).
_CUES: tuple[str, ...] = (
    r"\bincorrect\b",
    r"\bnot correct\b",
    r"\bwrong(ly)?\b",
    r"\berroneous(ly)?\b",
    r"\berror\b(?!\s*bars?)",
    r"\bmistake[sn]?\b",
    r"\btypo\b",
    r"\bmisnam",
    r"\bmislabel",
    r"\bmisidentif",
    r"\bshould (?:be|read)\b",
    r"\binconsisten",
    r"\bcontradict",
    r"\breversed\b",
    r"\breversal\b",
    r"\bimplausib",
    r"\bimpossib",
    r"\bfabricat",
    r"\bdoes not exist\b",
    r"\bnot (?:a )?(?:real|valid|known|traceable|verifiable)\b",
    r"\bcannot (?:be )?(?:find|found|locate[d]?|verif(?:y|ied))\b",
    r"\bcould not (?:be )?(?:find|found|locate[d]?|verif(?:y|ied))\b",
    r"\buntraceab",
    r"\bunverifiab",
    r"\binvalid\b",
    r"\bmisnomer\b",
)
# Case-insensitive because a finding capitalises its first word; nothing else about
# the alternation is relaxed.
ASSERTION_CUES = re.compile("|".join(_CUES), re.IGNORECASE)
NEGATION = re.compile(
    r"\b(no|not|never|without|none|nothing|neither|nor|unlikely|fails? to)\b", re.IGNORECASE
)
# A negation counts against a cue only inside the cue's own CLAUSE: from the sentence
# start, or from the last of these breaks before the cue. Commas are deliberately NOT
# breaks — "Dcr2, not Dcr1, is incorrect" would otherwise lose the negation that its
# own clause carries. The first cut of this rule read a fixed five words back instead,
# and missed "Nothing about the Dcr2 reference is incorrect" because "Nothing" sits six
# words before the cue; clause scope is the string-agnostic fix, and it errs by
# UNDER-crediting (see `asserts`), which is the safe direction for a published count.
CLAUSE_BREAKS = re.compile(r";|:|\s+but\s+|\s+however\s+|\s+although\s+|\s+whereas\s+", re.I)
# A citation parenthetical is one span of text, never a sentence boundary: the shared
# splitter cuts "(Hollingsworth and Vance, 2019, Nat. Metab. 7:e91188)" at "Nat." and
# "Metab.", which are not in its abbreviation list, and that would put the token in one
# sentence and the assertion about it in the next. So `asserts` — and ONLY `asserts` —
# masks parenthesised spans before splitting and restores them afterwards. The shared
# list in `text.chunks` is not touched: its boundaries key the extraction cache, and
# moving them would move every downstream artifact.
_PARENTHETICAL = re.compile(r"\([^()]*\)")
_MASKED_SPAN = re.compile(r"\(\x00(\d+)\x00\)")


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


def scored_text(text: str, quote: str) -> str:
    """A finding's `text`, with the `quote` field dropped — the FIRST of two steps.

    ONE function for both arms, and it discards the quote on purpose. A `quote`
    is by definition a verbatim slice of the manuscript, and the planted token
    is planted INTO the manuscript — so scoring text + quote credited an arm for
    reproducing the perturbed sentence, which is what every credited detection
    in the first committed records turned out to be (round-3 review, all three
    evaluators). The quote is still collected and committed, so a reader can see
    what each finding pointed at.

    This closes ONE of quotation's two routes, and until round 4 the published
    prose claimed it closed both. It does not: nothing stops a model putting
    manuscript text in `text`, and these models mostly do. `own_prose` below is
    the second step and the one that makes the claim true; read its docstring
    for the measurement.
    """
    del quote  # collected for the record, deliberately not scored
    return text.strip()


_WHITESPACE = re.compile(r"\s+")


def _normalised(text: str) -> str:
    """One shape for both sides of the quotation test: single spaces, case-folded."""
    return _WHITESPACE.sub(" ", text).strip().casefold()


# The punctuation a quotation picks up on its way into a finding, and nothing else. A
# model that copies a mid-paragraph clause and terminates it with a full stop writes a
# string that is not a substring of the manuscript, and round 5 measured what that costs
# (see `own_prose`). Both sets are plain characters, stripped as runs from the ends of
# the SENTENCE only — never from the manuscript, never from anywhere inside the
# sentence, and never per string. Nothing here is fuzzy: a word added to a quotation
# still survives, because only these characters are removed.
_QUOTATION_LEADING = "\"'([ "
_QUOTATION_TRAILING = ".,;:!?\"')] "


def _quotation_form(text: str) -> str:
    """`_normalised`, then enclosing quotes/brackets and terminal punctuation removed."""
    return _normalised(text).lstrip(_QUOTATION_LEADING).rstrip(_QUOTATION_TRAILING)


def own_prose(text: str, manuscript: str) -> str:
    """`text` with every sentence that is verbatim manuscript dropped — what v2 scores.

    The residue, not the raw string, is what `asserts` and `detect` are given. The
    text is cut on the SAME abbreviation-aware boundary the index uses
    (`peerpanel.text.chunks.sentences`, never a second splitter — DECISIONS D19); a
    sentence whose whitespace-normalised, case-folded form is a substring of the same
    normalisation of `manuscript` is dropped; what is left is joined back with single
    spaces. A finding that is entirely quotation returns `""`. A finding that appends
    a real assertion to a quotation keeps the assertion and loses the quotation.

    What the comparison tolerates, and why. Before the substring test the SENTENCE — never
    the manuscript — has a run of `.,;:!?"')]` taken off its end and a run of `"'([` off
    its front (`_quotation_form`). Nothing else: no fuzzy match, no similarity threshold,
    no per-string case. Terminal punctuation is not an exotic edge; it is the single most
    common difference between a copied clause and its source, and round 5 measured what
    tolerating it is worth on the committed records. Twenty-four sentences are quotation
    that the untolerant test kept, and the consequences reached two published numbers:
    the share of written characters that are manuscript slices moves from **59% to 68%**
    (29,325 of 43,416), the `single-agent-cot-sc:llama3.1:8b` arm's published "own prose,
    in characters — 41%" on caprin is really about **16%**, and the labelled `named token`
    bound on caprin falls from 2/3 to 1/3 for BOTH single-agent arms — two of the five
    credits on the README's first screen existed only because a copied sentence ended in
    a full stop. The `asserted` headline is 0 before and after, on every arm: the null is
    untouched, and what changed is the column published beside it. The enclosing-bracket
    half of the strip fires on no committed sentence today (all twenty-four are terminal
    punctuation); it is here because a quotation wrapped in quotes is the same class of
    string, and it is pinned by a control rather than left to a future run to discover.

    `manuscript` must be the PERTURBED text the arm was actually shown. Compared
    against the unperturbed original, every quotation of a planted sentence would
    survive — which is the exact string class this exists to remove.

    Why it exists. `scored_text` drops the `quote` field, and the repository published
    that as "only the finding's own prose is scored". Nothing stopped a model putting
    manuscript text in `text`, and the review that found this measured how often they
    do: **147 of the 235 scored strings — 63% — were verbatim slices of the perturbed
    manuscript** in the records it judged, where all 8 of one panel reviewer's met17
    findings had `text` byte-identical to their own `quote`, and every string that had
    ever earned a `named token` credit was one of them. The published `named token`
    column was therefore measuring quotation, not naming.

    Those records have since been superseded, and the figure moves with the run because
    it is a property of what the models wrote rather than of this code: read with the
    tolerance above, the records committed beside this file are **176 of 256 strings,
    69%** pure quotation, and **68% of the characters written back**. Every one of these
    numbers is recomputable — `tests/test_planted_soundness.py` measures the committed
    share on every test run and prints it per arm, in strings and in characters — and the
    ones quoted last are the ones a reader can check today.

    Why this is a fix and not a tune. It can only ever LOWER a count, never raise one:
    it only ever removes text before the two rules see it, and both are existential
    (some sentence of some string), so anything credited on the residue would have been
    credited on the raw string too. A post-hoc scoring change that is arithmetically
    incapable of flattering the headline is the one shape of post-hoc change that
    cannot be fitting to the result — which is why the cue list is untouched and only
    the input to it moved.

    Its limits, stated rather than hidden:

    - it is STILL a whole-sentence rule and STILL a substring test, and the punctuation
      tolerance narrows neither. A sentence that quotes the manuscript and adds three
      words of its own survives intact, and so does one that quotes with a typo, a
      reflowed dash, or a bracket dropped from the middle rather than the end. It
      under-removes, never over-removes, so a count carried through it is still an UPPER
      bound on non-quotation — the direction is unchanged by the repair, only the size of
      the gap is smaller than the previous docstring left a reader to assume;
    - the tolerated characters are stripped as runs from the ends, so a sentence that is
      punctuation alone normalises to nothing and is dropped. That removes no prose, and
      a short assertion is not at risk: "MET18 is incorrect." keeps every word it has and
      is only ever dropped if those words are themselves a slice of the manuscript, which
      is the substring rule doing its job rather than the strip overreaching;
    - it inherits the shared splitter's boundaries, including the single-character rule
      that keeps "conducted in R." joined to the sentence after it, so a quotation the
      splitter cuts differently from the manuscript's own layout is compared as the
      pieces it cut;
    - a sentence that merely shares vocabulary with the manuscript is NOT quotation and
      is kept. The test is substring, not similarity, on purpose: similarity would need
      a threshold, and a threshold is exactly the knob this rule must not have.
    """
    body = _normalised(manuscript)
    kept: list[str] = []
    for sentence in sentences(text):
        normalised = _quotation_form(sentence)
        if not normalised or normalised in body:
            continue
        kept.append(sentence.strip())
    return " ".join(kept)


_GENE = re.compile(r"\b([A-Z][a-z]{2}[0-9]{1,2}|[A-Z]{2,6}[0-9]{1,2})\b")
_PVALUE = re.compile(r"\bp\s*[<=]\s*0\.0*[0-9]+\b", re.IGNORECASE)
_PERCENT = re.compile(r"\b([0-9]{1,2}(?:\.[0-9])?)%")
_DIRECTION = re.compile(
    r"\b(increased|decreased|higher|lower|upregulated|downregulated)\b", re.IGNORECASE
)
_FLIP = {
    "increased": "decreased",
    "decreased": "increased",
    "higher": "lower",
    "lower": "higher",
    "upregulated": "downregulated",
    "downregulated": "upregulated",
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
        wrong = symbol[: digits.start()] + str(int(digits.group()) + 1) if digits else symbol + "2"
        _apply(i, symbol, wrong, "gene_symbol_swap", wrong)

    # 2. Effect-direction flip: the classic reviewer catch.
    directions = [
        (i, m.group(1))
        for i, line in enumerate(lines[:plantable])
        for m in [_DIRECTION.search(line)]
        if m
    ]
    for i, word in rng.sample(directions, min(n_per_kind, len(directions))):
        flipped = _FLIP[word.lower()]
        # The detection token is a PHRASE pinned to the perturbed sentence, not the
        # bare flipped word: "decreased" alone occurs in 37 of the 68 demo-corpus
        # documents, so retrieved literature quoted back by either arm could mint a
        # catch nobody earned. The phrase is the flipped word plus the two words
        # that follow it in the sentence it was planted in.
        after = lines[i].split(word, 1)[1].split() if word in lines[i] else []
        token = " ".join([flipped, *after[:2]]) if after else flipped
        _apply(i, word, flipped, "effect_direction_flip", token)

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
            "gene_symbol_swap",
            "effect_direction_flip",
            "impossible_pvalue",
            "fabricated_citation",
            "impossible_percentage",
        )
        if kind not in planted_kinds
    }
    return PlantedManuscript(
        source_file=source_file,
        text="\n".join(lines),
        errors=errors,
        skipped_kinds=skipped,
    )



def _mask_parentheticals(text: str) -> tuple[str, list[str]]:
    """Replace each parenthesised span with a period-free placeholder of the same shape.

    The placeholder keeps the outer brackets, so a sentence that BEGINS with a
    parenthetical still starts where it did — the splitter's lookahead admits "(" on
    purpose. Nested or unbalanced brackets are left exactly as they are: masking them
    would need a matcher this rule does not earn, and the plain splitter is then what
    scores the text, with the same limits it has everywhere else.
    """
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
            if depth > 1:
                return text, []  # nested — leave it to the plain splitter
        elif char == ")":
            depth -= 1
            if depth < 0:
                return text, []  # unbalanced — same
    if depth != 0:
        return text, []
    spans: list[str] = []

    def _swap(match: re.Match[str]) -> str:
        spans.append(match.group(0))
        return f"(\x00{len(spans) - 1}\x00)"

    return _PARENTHETICAL.sub(_swap, text), spans


def _assertion_sentences(text: str) -> list[str]:
    """The sentences `asserts` scores: the shared boundary, with citations kept whole."""
    masked, spans = _mask_parentheticals(text)
    if not spans:
        return sentences(text)
    return [
        _MASKED_SPAN.sub(lambda m: spans[int(m.group(1))], sentence)
        for sentence in sentences(masked)
    ]


def _is_negated(sentence: str, cue_start: int) -> bool:
    """Does a negation stand between the start of the cue's clause and the cue?

    A cue that carries its own negation ("does not exist", "not correct", "cannot be
    verified") matches as a whole phrase beginning AT that negation, so the negation is
    inside the cue and never in the text scanned in front of it.
    """
    clause_start = 0
    for break_ in CLAUSE_BREAKS.finditer(sentence, 0, cue_start):
        clause_start = break_.end()
    return NEGATION.search(sentence[clause_start:cue_start]) is not None


def asserts(error: PlantedError, finding_texts: list[str]) -> bool:
    """Did some finding ASSERT the defect this planted error records, in its own prose?

    The rule (`DETECTION_RULE`, "assertion-v2"): a finding is credited iff ONE of its
    sentences — cut on the abbreviation-aware boundary the index uses
    (`peerpanel.text.chunks.sentences`, never a second splitter), with parenthesised
    spans held whole — contains both the detection token (case-insensitive substring,
    exactly as `detect`) and an assertion cue from `ASSERTION_CUES` that no negation in
    the cue's own clause stands in front of.

    It errs in BOTH directions, and a published number must carry both with it:

    - it does not credit a finding that asserts the defect without naming the token
      ("the direction of this effect is reversed" earns nothing for
      "decreased accumulation of"), so the count is a FLOOR on detection;
    - it does credit a finding that names the token and uses a cue about something
      else in the same sentence ("Dcr2 is discussed, and the error in Figure 2 is
      obvious" counts for the Dcr1->Dcr2 swap), so the floor is not a clean one;
    - a negation anywhere earlier in the cue's clause suppresses the cue, so
      "Dcr2, not Dcr1, is incorrect" — a real assertion — is NOT credited: clause scope
      cannot tell a negated claim from a corrected one, and it under-credits rather
      than over-credits on purpose;
    - it is same-sentence only, so an assertion split across two sentences ("Dcr2 is
      named here. That symbol is incorrect.") earns nothing.

    Cues that occur in these manuscripts as ordinary vocabulary were dropped before
    any control was run, and `error` keeps an `error bars` exception, so that quoted
    methods prose cannot mint a catch: `invert` (inverted microscope), `opposite`,
    bare `revers` (reverse transcribed), `no such` (no such enrichment was observed),
    `suspicious`, `questionable`.

    Calibrated before first use. The first cut of the mechanism read a fixed five words
    back for a negation and split sentences with the shared boundary untouched. Two
    pre-run controls defeated it, before it had scored any record: "Nothing about the
    Dcr2 reference is incorrect." was CREDITED (the negation sits six words before the
    cue), and "The citation (Hollingsworth and Vance, 2019, Nat. Metab. 7:e91188) does
    not exist." was NOT credited (the splitter cut the citation at "Nat." and "Metab.",
    leaving the token in one sentence and the assertion in the next). The cue list was
    not touched; the two mechanisms around it were replaced with clause-scoped negation
    and masked parentheticals, both string-agnostic. `DETECTION_RULE` keeps its name
    because no record was ever produced by the pre-calibration code.

    What reaches it is the v2 change and the only one: the evaluation passes the
    `own_prose` residue of each string, not the raw string, so a sentence that is
    verbatim manuscript is gone before this function is called. The judgement below is
    v1's, unchanged, and the substitution can only remove candidate sentences — see
    `own_prose` for why that direction is the whole argument.

    Every string this rule scored is committed beside the number it produced, in the
    record's `finding_texts` (everything the arm wrote) and `scored_texts` (the residue
    actually scored), so a reader can judge each decision rather than trust the count.
    `detect` stays beside it as the "named the token" upper bound.
    """
    token = error.detection_token.lower()
    for text in finding_texts:
        for sentence in _assertion_sentences(text):
            if token not in sentence.lower():
                continue
            for cue in ASSERTION_CUES.finditer(sentence):
                if not _is_negated(sentence, cue.start()):
                    return True
    return False



def detect(error: PlantedError, finding_texts: list[str]) -> bool:
    """Did some finding NAME the planted token, in its own prose?

    The rule is a case-insensitive substring test, and it errs in BOTH
    directions, which the published numbers must carry with them:

    - it under-credits a finding that describes the problem without naming the
      token ("the reported significance is impossible" does not count for
      `p = 1.34`);
    - it over-credits a finding that names the token while merely restating the
      manuscript ("the excerpt mentions Dicer (Dcr2)" counts for a Dcr1->Dcr2
      swap, though it asserts nothing wrong). A substring rule cannot tell an
      assertion from a restatement, and no mechanical rule here tries to.

    So the number this returns is "named the token", an UPPER bound on
    detection rather than a floor. The strings it credited are committed with
    every record so a reader can judge each one; the round-3 review did, and
    found that on the first records none asserted an error.
    """
    token = error.detection_token.lower()
    return any(token in text.lower() for text in finding_texts)


def write_planted(path: Path, planted: PlantedManuscript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(planted.model_dump_json(indent=1) + "\n")
