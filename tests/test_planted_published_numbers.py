"""The planted evaluation's published numbers, bound to the records STRUCTURALLY.

Round 3's defect class was a guard that could not fail: a binding that asserts a
string appears *somewhere* in a file passes while the row it was meant to guard says
something else entirely. So nothing here searches for a value. Every table is parsed
into `{row label: {column: cell}}`, every cell is located by ROW LABEL and COLUMN
HEADER, and the cell found there is compared with the record. A table that lost a row,
renamed an arm or dropped a column fails on the SHAPE before any number is read.

The shape both tables must carry (D-2 / D-7):

- one row per arm — `panel`, and `single agent` naming each model in
  `planted_eval.BASELINE_MODELS`; the row label names the model, because two rows that
  differ only in a wire cannot report two models' pass rates;
- per manuscript, four columns whose headers name the manuscript and the measure:
  `asserted` (the headline, produced by the rule `planted.DETECTION_RULE` names),
  `named token` (the labelled upper bound), `tokens` and `wall` — e.g.
  `| caprin asserted | caprin named token | caprin tokens | caprin wall |`.

`results/RESULTS.md` must carry EVERY committed record; the README's first screen must
carry at least one, and every manuscript it does carry is bound the same way. Header
and label matching is normalised (case, emphasis, backticks, whitespace) and never
positional, so the prose keeps its formatting freedom and none of its numbers.

Moved out of `test_published_numbers.py`, which two owners were editing at once.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import NamedTuple

import pytest

from _prose_rules import LOSS_WORDS, WIN_WORDS
from peerpanel.evals.planted_eval import BASELINE_MODELS, baseline_system

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
# The first screen: requirement 10 scopes it to lines 1-60, so the binding does too.
HEAD = "\n".join(README.splitlines()[:60])
RESULTS = (ROOT / "results" / "RESULTS.md").read_text()

PANEL_ROW = "panel"
# Repeat runs (`-run2`) are a second measurement of the same subject, not a second
# subject, and would collide on the manuscript key; the published tables report run 1.
_REPEAT = re.compile(r"-run\d+$")


def _records() -> dict[str, dict[str, object]]:
    """Every committed planted-eval record, keyed by manuscript subject."""
    found: dict[str, dict[str, object]] = {}
    for path in sorted((ROOT / "results").glob("planted-eval-*.json")):
        subject = path.stem.removeprefix("planted-eval-")
        if _REPEAT.search(subject):
            continue
        found[subject] = json.loads(path.read_text())
    assert found, "no results/planted-eval-*.json records to bind the tables to"
    return found


def _key(subject: str) -> str:
    """The short name a table column may use for a manuscript: `caprin`, `met17`."""
    return subject.split("-")[0].lower()


def _arm(record: dict[str, object], system: str) -> dict[str, object]:
    results = record["results"]
    assert isinstance(results, list)
    matches = [r for r in results if r["system"] == system]
    assert len(matches) == 1, (
        f"{record['manuscript']}: {len(matches)} arms named {system!r}; the record must "
        f"carry exactly one — it has {[r['system'] for r in results]}"
    )
    arm: dict[str, object] = matches[0]
    return arm


def _norm(cell: str) -> str:
    """A table cell as text: emphasis, backticks and runs of whitespace are formatting."""
    return re.sub(r"\s+", " ", re.sub(r"[*`_]", "", cell)).strip().lower()


def _tables(text: str) -> list[list[list[str]]]:
    """Every markdown table in `text`, as a list of rows of raw cells."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            current.append([c.strip() for c in stripped.strip("|").split("|")])
            continue
        if current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return [t for t in tables if len(t) >= 3]


def _is_separator(row: list[str]) -> bool:
    return all(re.fullmatch(r":?-{2,}:?", c.strip()) is not None for c in row)


def _parse(table: list[list[str]]) -> tuple[list[str], dict[str, dict[str, str]]]:
    """(normalised header cells, {row label -> {column header -> cell}})."""
    header = [_norm(c) for c in table[0]]
    body = table[2:] if _is_separator(table[1]) else table[1:]
    rows: dict[str, dict[str, str]] = {}
    for row in body:
        label = _norm(row[0])
        rows[label] = dict(zip(header[1:], row[1:], strict=False))
    return header, rows


def _is_baseline_label(label: str) -> bool:
    return "single agent" in label or "single-agent" in label


def _row_for(rows: dict[str, dict[str, str]], arm: str) -> dict[str, str]:
    """The row a labelled arm occupies. `arm` is 'panel' or a model id.

    The panel row is expected to NAME ITS MIXTURE (`panel (llama3.1:8b + qwen2:7b)`),
    so a model name cannot be what separates the rows — being a single-agent row is.
    """
    if arm == PANEL_ROW:
        wanted = [label for label in rows if "panel" in label and not _is_baseline_label(label)]
    else:
        wanted = [
            label for label in rows if _is_baseline_label(label) and arm.lower() in label
        ]
    assert len(wanted) == 1, (
        f"expected exactly one row for {arm!r}; found {wanted} among {sorted(rows)}. "
        f"The table must carry one row per arm: {PANEL_ROW}, and 'single agent' naming "
        f"each of {list(BASELINE_MODELS)}."
    )
    return rows[wanted[0]]


_MEASURES = ("asserted", "named token", "tokens", "wall")


def _measure_of(header: str) -> str | None:
    """Which measure a column header names — `named token` is checked first, since it
    also contains the word the tokens column is named for."""
    if "named" in header:
        return "named token"
    if "assert" in header:
        return "asserted"
    if "token" in header:
        return "tokens"
    if "wall" in header:
        return "wall"
    return None


def _columns(header: list[str], subject: str) -> dict[str, str]:
    """{measure -> column header} for one manuscript's group of columns."""
    key = _key(subject)
    found: dict[str, str] = {}
    for cell in header[1:]:
        if key not in cell and subject.lower() not in cell:
            continue
        measure = _measure_of(cell)
        if measure is None:
            continue
        assert measure not in found, (
            f"two columns claim to be {subject}'s {measure!r}: {found[measure]!r} and {cell!r}"
        )
        found[measure] = cell
    return found


def _count(cell: str, expected: int, n_errors: int, where: str) -> None:
    text = _norm(cell)
    match = re.search(r"(\d+)\s*(?:/\s*(\d+))?", text)
    assert match, f"{where}: {cell!r} carries no count"
    assert int(match.group(1)) == expected, (
        f"{where}: the table says {cell!r}; the record says {expected} of {n_errors}"
    )
    if match.group(2) is not None:
        assert int(match.group(2)) == n_errors, (
            f"{where}: the table's denominator is {match.group(2)}; the record planted "
            f"{n_errors} errors"
        )


def _tokens(cell: str, expected: int, where: str) -> None:
    assert f"{expected:,}" in cell, (
        f"{where}: the table says {cell!r}; the record's total_tokens is {expected:,}"
    )


_UNITS = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "m": 60.0, "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
}


def _wall(cell: str, expected: float, where: str) -> None:
    """The wall cell, in whichever unit it is published, to its own precision."""
    text = _norm(cell)
    match = re.search(r"([\d,]+(?:\.\d+)?)\s*([a-z]+)?", text)
    assert match, f"{where}: {cell!r} carries no wall-clock figure"
    shown = float(match.group(1).replace(",", ""))
    unit = (match.group(2) or "s").strip()
    assert unit in _UNITS, (
        f"{where}: {cell!r} names the unit {unit!r}; wall-clock is published in "
        f"{sorted(set(_UNITS))}"
    )
    decimals = len(match.group(1).split(".")[1]) if "." in match.group(1) else 0
    converted = expected / _UNITS[unit]
    # Half of the cell's own last digit: the published figure must round to the record
    # at whatever precision it chose. An `==` against round() would fail on a value
    # sitting exactly on a rounding boundary, where both neighbours are honest.
    slack = 0.5 * 10.0**-decimals + 1e-9
    assert abs(shown - converted) <= slack, (
        f"{where}: the table says {cell!r}; the record's wall_s is {expected} "
        f"({converted:.3f} {unit})"
    )


def _bind_table(table: list[list[str]], subjects: dict[str, dict[str, object]], where: str) -> int:
    """Every arm x manuscript cell in one table, against the records. Returns the
    number of manuscripts the table actually carried."""
    header, rows = _parse(table)
    arms = [PANEL_ROW, *BASELINE_MODELS]
    assert len(rows) == len(arms), (
        f"{where}: the table has {len(rows)} rows ({sorted(rows)}); the record has "
        f"{len(arms)} arms ({arms}) — a table that cannot show every arm cannot report "
        "a pass rate per model"
    )
    bound = 0
    for subject, record in subjects.items():
        columns = _columns(header, subject)
        if not columns:
            continue
        missing = [m for m in _MEASURES if m not in columns]
        assert not missing, (
            f"{where}: {subject}'s columns are missing {missing} (found {columns}). The "
            "headline (asserted) and its labelled upper bound (named token) are both "
            "published, with cost as tokens and wall-clock."
        )
        n_errors = int(record["n_errors"])  # type: ignore[call-overload]
        for arm in arms:
            system = PANEL_ROW if arm == PANEL_ROW else baseline_system(arm)
            measured = _arm(record, system)
            row = _row_for(rows, arm)
            place = f"{where} · {subject} · {arm}"
            for measure in _MEASURES:
                assert columns[measure] in row, f"{place}: no {measure!r} cell in the row"
            _count(
                row[columns["asserted"]],
                len(measured["detected"]),  # type: ignore[arg-type]
                n_errors,
                f"{place} · asserted",
            )
            _count(
                row[columns["named token"]],
                len(measured["named_token"]),  # type: ignore[arg-type]
                n_errors,
                f"{place} · named token",
            )
            _tokens(
                row[columns["tokens"]],
                int(measured["total_tokens"]),  # type: ignore[call-overload]
                f"{place} · tokens",
            )
            _wall(
                row[columns["wall"]],
                float(measured["wall_s"]),  # type: ignore[arg-type]
                f"{place} · wall",
            )
        bound += 1
    return bound


def _eval_table(text: str, where: str) -> list[list[str]]:
    """The one table carrying the panel row and every baseline arm."""
    candidates = []
    for table in _tables(text):
        _header, rows = _parse(table)
        labels = list(rows)
        if not any("panel" in label and not _is_baseline_label(label) for label in labels):
            continue
        if not all(
            any(_is_baseline_label(label) and m.lower() in label for label in labels)
            for m in BASELINE_MODELS
        ):
            continue
        candidates.append(table)
    assert len(candidates) == 1, (
        f"{where}: found {len(candidates)} evaluation tables carrying a panel row and a "
        f"row for each of {list(BASELINE_MODELS)}; there must be exactly one"
    )
    return candidates[0]


class TestEveryRecordIsPublished:
    def test_every_committed_eval_record_is_discussed(self) -> None:
        for subject in _records():
            assert subject in RESULTS, (
                f"planted-eval-{subject}.json is committed but {subject} is never "
                "mentioned in RESULTS.md — a measurement nobody reads is a measurement "
                "nobody checked"
            )


class TestTheThreeArmTableMatchesItsRecords:
    """Each published cell, located by row label and column header, against the JSON."""

    def test_the_results_table_binds_every_manuscript(self) -> None:
        subjects = _records()
        bound = _bind_table(
            _eval_table(RESULTS, "results/RESULTS.md"), subjects, "results/RESULTS.md"
        )
        assert bound == len(subjects), (
            f"results/RESULTS.md's three-arm table carries {bound} of "
            f"{len(subjects)} committed manuscripts ({sorted(subjects)}); every "
            "committed record is published there"
        )

    def test_the_first_screen_table_binds_what_it_carries(self) -> None:
        subjects = _records()
        bound = _bind_table(
            _eval_table(HEAD, "README.md lines 1-60"), subjects, "README.md lines 1-60"
        )
        assert bound >= 1, (
            "the first screen's evaluation table names none of the committed "
            f"manuscripts ({sorted(subjects)}) in its column headers, so not one of its "
            "numbers could be bound to a record"
        )


def _slack(shown: str) -> float:
    """Half of the last digit the claim published — the honest rounding window.

    The rule this repository already states for its other published ratio, applied
    here word for word (`tests/test_published_numbers.py::_slack`): "6.6x" must round
    to the record's ratio one digit finer than "7x" would. A fixed tolerance is how a
    published 4x survived a measured 3.2x — and how a fixed +/-0.2 on these multiples
    accepted a claim about the wrong manuscript (round 4, report 3, finding 5).
    """
    decimals = len(shown.split(".")[1]) if "." in shown else 0
    return 0.5 * 10.0**-decimals + 1e-9


# How a reader resolves a cost sentence that does not repeat the manuscript's name.
# These are the ONLY two shortcuts read; anything else unbound fails and asks for the
# manuscript by name, so the resolver can never quietly invent a subject.
_ORDINALS = {"first": 0, "second": 1, "third": 2}
_CONTINUATIONS = ("the same run", "the same manuscript", "that run", "the same one")


def _paragraph_bounds(text: str, at: int) -> tuple[int, int]:
    start = text.rfind("\n\n", 0, at)
    end = text.find("\n\n", at)
    return (0 if start == -1 else start + 2, len(text) if end == -1 else end)


def _sentence_bounds(text: str, at: int) -> tuple[int, int]:
    """The sentence around `at`, never crossing its paragraph."""
    para_start, para_end = _paragraph_bounds(text, at)
    breaks = (". ", ".\n", "! ", "? ")
    starts = [text.rfind(end, para_start, at) for end in breaks]
    ends = [text.find(end, at, para_end) for end in breaks]
    return (
        max([s + 2 for s in starts if s != -1], default=para_start),
        min([e + 1 for e in ends if e != -1], default=para_end),
    )


def _named_in(scope: str, subjects: list[str]) -> list[str]:
    """Which manuscripts a span of prose names, by full name or short key."""
    low = scope.lower()
    return [s for s in subjects if s.lower() in low or _key(s) in low]


def _subject_order(scope: str, subjects: list[str]) -> list[str]:
    """The manuscripts a passage names, in the order it first names them.

    What "the second" means to a reader is the second manuscript THIS passage has
    discussed, so the order is read from the claim's own paragraph and never from the
    whole document (where an unrelated earlier mention would silently reverse it).
    Manuscripts the passage never names are not in the list, so an ordinal that points
    past its end fails instead of guessing.
    """
    low = scope.lower()

    def first(subject: str) -> int:
        hits = [i for i in (low.find(subject.lower()), low.find(_key(subject))) if i != -1]
        return min(hits) if hits else len(scope)

    return [s for s in sorted(subjects, key=first) if first(s) < len(scope)]


class Claim(NamedTuple):
    """One published cost claim: what it says, about which arm, on which manuscript.

    `shown` is the literal digits as published, so the rounding window is read from the
    claim's own precision; `subject` is the manuscript its sentence names, because a
    multiple is true of one manuscript and false of the other.
    """

    model: str
    subject: str
    kind: str
    shown: str
    at: int
    where: str


class TestTheCostClaims:
    """A multiplier or share sentence must say WHICH baseline it compares against and
    WHICH manuscript it is about: with one baseline arm and one manuscript "6.7x the
    tokens" was unambiguous; with two of each it is a number a reader cannot check.

    Round 4 (report 3, finding 5) demonstrated what the old guard let through. It
    accepted a multiple matching ANY committed record, on a fixed +/-0.2, so

        "The panel spent 7.9x the tokens of the llama3.1:8b single agent on caprin"

    PASSED on met17's 7.85 while being wrong for caprin by 1.3x — the "a guard that
    asserts a string appears SOMEWHERE" class this file exists to keep out. Both halves
    of the fix are below: the manuscript is attributed the way the model already was,
    and the tolerance is this repository's own rounding rule (`_slack`), not a fixed
    window it has ruled insufficient in writing.

    Attribution reads the prose the way a reader does, in this order, from the claim's
    own side of the sentence outwards: the manuscript named after the claim (up to the
    next claim), then before it (back to the previous one), then in its sentence, then
    in its paragraph. A span that names exactly one manuscript binds. Failing a name,
    an ordinal ("the second") indexes the manuscripts in the order this document names
    them, and a continuation ("the same run") carries the previous claim's manuscript
    forward. A claim none of those resolve FAILS, naming what to write instead.
    """

    MULTIPLIER = re.compile(
        r"(\d+(?:\.\d+)?)\s*[x\u00d7]\s*the\s+(tokens|wall-clock|wall clock)", re.IGNORECASE
    )
    # Any percentage in a sentence that states a share of the panel is a share claim —
    # "16% of the panel's tokens on the same manuscript and 12% on the second" publishes
    # TWO numbers, and a pattern that reads only the first leaves the second bound by
    # nothing, which is the same defect one clause later.
    SHARE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
    SHARE_SENTENCE = re.compile(r"\d+(?:\.\d+)?\s*% of the panel", re.IGNORECASE)

    # The evaluator's own demonstration, kept live as a control: the first sentence must
    # be rejected (7.9 is met17's multiple, not caprin's) and the second accepted.
    WRONG_SENTENCE = (
        "The panel spent 7.9x the tokens of the llama3.1:8b single agent on "
        "caprin-heterochromatin."
    )
    RIGHT_SENTENCE = (
        "The panel spent 6.6x the tokens of the llama3.1:8b single agent on "
        "caprin-heterochromatin."
    )

    def _claims(self, text: str, where: str) -> list[Claim]:
        """Every cost claim in `text`, each bound to a model and a manuscript.

        `kind` is one of `tokens`, `wall-clock` (a multiplier of the panel's) or
        `share` (a percentage of the panel's tokens).
        """
        subjects = sorted(_records())
        pairs: list[tuple[re.Match[str], str]] = [
            (match, match.group(2).lower().replace(" ", "-"))
            for match in self.MULTIPLIER.finditer(text)
        ]
        for match in self.SHARE.finditer(text):
            start, end = _sentence_bounds(text, match.start())
            if self.SHARE_SENTENCE.search(text[start:end]):
                pairs.append((match, "share"))
        matches = sorted(pairs, key=lambda pair: pair[0].start())
        spans = [match.span() for match, _kind in matches]
        found: list[Claim] = []
        for index, (match, kind) in enumerate(matches):
            model = self._attribute(text, match.start(), match.group(0), where)
            subject = self._subject(text, spans, index, subjects, found, match.group(0), where)
            found.append(Claim(model, subject, kind, match.group(1), match.start(), where))
        return found

    def _attribute(self, text: str, at: int, claim: str, where: str) -> str:
        """Which baseline arm a claim is about: the model named in its own sentence,
        or failing that in its own paragraph."""
        sentence_start, sentence_end = _sentence_bounds(text, at)
        paragraph_start, paragraph_end = _paragraph_bounds(text, at)
        for scope in (
            text[sentence_start:sentence_end],
            text[paragraph_start:paragraph_end],
        ):
            named = [m for m in BASELINE_MODELS if m.lower() in scope.lower()]
            if len(named) == 1:
                return named[0]
        raise AssertionError(
            f"{where}: the claim {claim!r} names no single baseline model in its sentence "
            f"or its paragraph, so a reader cannot tell which of {list(BASELINE_MODELS)} "
            "it compares the panel against. Name the model in the same sentence."
        )

    def _subject(
        self,
        text: str,
        spans: list[tuple[int, int]],
        index: int,
        subjects: list[str],
        earlier: list[Claim],
        claim: str,
        where: str,
    ) -> str:
        """Which manuscript a claim is about — the same discipline as `_attribute`.

        Windows are read innermost first (see the class docstring); a window naming two
        manuscripts is ambiguous, not a match, so the next window is tried.
        """
        start, end = spans[index]
        sentence_start, sentence_end = _sentence_bounds(text, start)
        paragraph_start, paragraph_end = _paragraph_bounds(text, start)
        after = next((s for s, _e in spans[index + 1 :] if s >= end), sentence_end)
        before = next((e for _s, e in reversed(spans[:index]) if e <= start), sentence_start)
        windows = (
            text[end : max(end, min(after, sentence_end))],
            text[min(start, max(before, sentence_start)) : start],
            text[sentence_start:sentence_end],
            text[paragraph_start:paragraph_end],
        )
        for scope in windows:
            named = _named_in(scope, subjects)
            if len(named) == 1:
                return named[0]
            if named:
                continue
            ordinals = [word for word in _ORDINALS if f"the {word}" in scope.lower()]
            if len(ordinals) == 1:
                position = _ORDINALS[ordinals[0]]
                order = _subject_order(text[paragraph_start:paragraph_end], subjects)
                assert position < len(order), (
                    f"{where}: the claim {claim!r} points at 'the {ordinals[0]}' manuscript, "
                    f"but its paragraph names {len(order)} ({order}). Name the manuscript."
                )
                return order[position]
            if any(phrase in scope.lower() for phrase in _CONTINUATIONS):
                assert earlier and paragraph_start <= earlier[-1].at < paragraph_end, (
                    f"{where}: the claim {claim!r} carries its manuscript forward from a "
                    "previous claim, but no earlier cost claim stands in the same "
                    "paragraph. Name the manuscript in the sentence."
                )
                return earlier[-1].subject
        raise AssertionError(
            f"{where}: the claim {claim!r} names no single manuscript in its own clause, "
            f"sentence or paragraph, so a reader cannot tell which of {subjects} it is "
            "about — and a multiple true of one is false of the other. Name the "
            "manuscript in the same clause (or say 'the same run' to carry the previous "
            "claim's manuscript forward)."
        )

    def _all(self) -> list[Claim]:
        return self._claims(RESULTS, "results/RESULTS.md") + self._claims(README, "README.md")

    def _record_values(self, claim: Claim) -> dict[str, float]:
        """{manuscript: what the record gives for this claim's arm and measure}."""
        system = baseline_system(claim.model)
        values: dict[str, float] = {}
        for subject, record in _records().items():
            panel = _arm(record, PANEL_ROW)
            arm = _arm(record, system)
            if claim.kind == "tokens":
                values[subject] = float(panel["total_tokens"]) / float(arm["total_tokens"])  # type: ignore[arg-type]
            elif claim.kind == "share":
                values[subject] = (
                    float(arm["total_tokens"]) / float(panel["total_tokens"]) * 100  # type: ignore[arg-type]
                )
            else:  # wall-clock
                values[subject] = float(panel["wall_s"]) / float(arm["wall_s"])  # type: ignore[arg-type]
        return values

    def _check(self, claims: list[Claim]) -> None:
        for claim in claims:
            values = self._record_values(claim)
            assert claim.subject in values, (
                f"{claim.where}: the claim is about {claim.subject}, which has no "
                f"committed record ({sorted(values)})"
            )
            unit = "%" if claim.kind == "share" else "x"
            slack = _slack(claim.shown)
            assert float(claim.shown) == pytest.approx(values[claim.subject], abs=slack), (
                f"{claim.where}: publishes {claim.shown}{unit} ({claim.kind}) for "
                f"{claim.model} on {claim.subject}, where the record gives "
                f"{values[claim.subject]:.2f} (the rounding window this claim's own "
                f"precision allows is +/-{slack:.3g}). Every record: "
                + ", ".join(f"{s} {v:.2f}" for s, v in sorted(values.items()))
                + ". A multiple that matches a DIFFERENT manuscript is the defect this "
                "guard exists for."
            )

    def test_every_cost_claim_matches_a_record(self) -> None:
        self._check(self._all())

    def test_the_guard_reads_the_manuscript_the_sentence_names(self) -> None:
        """Round 4's demonstration, kept as a control that can go red.

        A guard is worth what its failure proves, so the sentence that passed the old
        binding is checked here against the manuscript it names.
        """
        wrong = self._claims(self.WRONG_SENTENCE, "<control>")
        assert [(c.model, c.subject, c.kind, c.shown) for c in wrong] == [
            ("llama3.1:8b", "caprin-heterochromatin", "tokens", "7.9")
        ], f"the control sentence did not parse as one caprin claim about llama: {wrong}"
        with pytest.raises(AssertionError, match="caprin-heterochromatin"):
            self._check(wrong)
        self._check(self._claims(self.RIGHT_SENTENCE, "<control>"))

    def test_the_rounding_rule_is_this_repositorys_one_rule(self) -> None:
        """`_slack` here is the rule stated in `test_published_numbers.py`, not a second
        one that happens to agree today: two tolerances for one written standard is how
        the standard drifts (round 4's shape — a description coming apart from what it
        describes)."""
        from test_published_numbers import _slack as latency_slack

        for shown in ("4", "3.2", "6.57", "15", "0.125"):
            assert _slack(shown) == latency_slack(shown), (
                f"the cost guard and the latency guard round {shown!r} differently"
            )

    def test_every_baseline_arm_carries_a_cost_claim(self) -> None:
        """The guard is proven live, per arm: a regex that matches nothing is round 3's
        defect class, and an arm nobody compares is an arm nobody reads."""
        named = {claim.model for claim in self._all()}
        missing = [m for m in BASELINE_MODELS if m not in named]
        assert not missing, (
            f"no cost claim in README.md or results/RESULTS.md is attributed to {missing}. "
            "Each baseline arm needs one sentence naming its model and stating the panel's "
            "cost against it — e.g. 'the panel spent 6.7x the tokens of the llama3.1:8b "
            "baseline on caprin' or 'qwen2:7b used 15% of the panel's tokens on met17'."
        )

    def test_every_manuscript_carries_a_cost_claim(self) -> None:
        """The subject binding is proven live too: if every published claim resolved to
        one manuscript, the attribution above would be guarding nothing."""
        named = {claim.subject for claim in self._all()}
        missing = [s for s in _records() if s not in named]
        assert not missing, (
            f"no cost claim in README.md or results/RESULTS.md is about {missing}, so the "
            "manuscript each claim names is never exercised on the real pages. Every "
            "committed manuscript's cost is published."
        )

class TestTheResultSentenceFollowsTheRecord:
    def _panel_wins_anywhere(self) -> bool:
        for record in _records().values():
            panel = len(_arm(record, PANEL_ROW)["detected"])  # type: ignore[arg-type]
            baselines = [
                len(_arm(record, baseline_system(m))["detected"])  # type: ignore[arg-type]
                for m in BASELINE_MODELS
            ]
            if panel > max(baselines):
                return True
        return False

    def test_the_word_guards_are_live(self) -> None:
        """Both regexes are asserted to be absent from the real file below, so their
        controls are literals: a dead regex would otherwise prove the prose honest."""
        assert WIN_WORDS.search("the panel beats the single agent on both manuscripts")
        assert LOSS_WORDS.search("the headline is a loss")

    def test_the_result_sentence_follows_the_record(self) -> None:
        if self._panel_wins_anywhere():
            pytest.skip(
                "a manuscript's panel asserted more than every baseline — the win-word "
                "rule only binds a record that shows no win"
            )
        for text, where in ((HEAD, "README.md lines 1-60"), (RESULTS, "results/RESULTS.md")):
            hits = [m.group(0) for m in WIN_WORDS.finditer(text)]
            assert not hits, (
                f"{where} claims the panel won ({hits}); no committed record shows the "
                "panel asserting more than every baseline arm"
            )
        assert LOSS_WORDS.search(HEAD), (
            "no committed record shows the panel winning, and the first screen never "
            "says so: requirement 10 wants the result as the record shows it — a loss "
            "or a null result, never a win"
        )
