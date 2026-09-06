"""The README's first screen — requirement 10, checked STRUCTURALLY.

A stranger gives this repository ninety seconds and reads sixty lines. Requirement 10
says what those lines must carry: the evaluation table with the assertion-only count
and the cost per named local model, the panel-vs-single result *as the record shows
it*, the sentence that this is a demonstration system, a link to the credibility map,
and — for the review loop, whose record is private — the round count, the findings
total, how many rounds came back clean and the pinned evaluator's sha256. Every number
there is read from a named `results/` record by this file.

Structurally, and the word is load-bearing. Round 3's defect class was a guard that
asserted a string appears SOMEWHERE in a document: it passes while the row it was
written to guard says something else entirely. So the table is parsed into
`{row label: {column header: cell}}`, every cell is located by row label and column
header, and the shape is asserted before any number is read — a table that loses an
arm, renames a column or drops a manuscript fails here, loudly.

`test_planted_published_numbers.py` binds the same table from the record's side. The
two parsers are deliberately independent: they agree by measuring the same file, not by
sharing code, which is the twin-check shape this repo uses wherever a number matters.
The one thing they do share is `tests/_prose_rules.py`, because a result-sentence rule
enforced by two different word lists is not one rule.

Red until the first screen is rewritten on the regenerated records (CP3) — it collects,
and every failure names the file, the cell and both values.

Below the table bindings, this file also binds the figures the page RESTATES elsewhere
— the exclusion residue on the first screen, the index counts in the diagram and in the
two-scales paragraph, and the grounding yield. The promise on line 4 is made about the
whole page, not about the first screen, and round 5 moved ten of those figures with the
suite green; the voice sweep at the end of this file already reads the whole README for
the same reason. Each of those rules is a function over the page's text with a control
beside it that hands it a mutated page and proves it goes red.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from _prose_rules import LOSS_WORDS, WIN_WORDS
from peerpanel.evals.planted_eval import BASELINE_MODELS, PANEL_SYSTEM, baseline_system

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"
README = README_PATH.read_text()
# Requirement 10 scopes the first screen to the first 60 lines, none over 160 characters
# — the same window checklist line 1 verifies with `head -60 README.md | awk 'length >
# 160'`. HEAD is that window, and nothing below it is the first screen.
HEAD_LINES = README.splitlines()[:60]
HEAD = "\n".join(HEAD_LINES)
MAX_LINE = 160

MEASURES = ("asserted", "named token", "tokens", "wall")
# A repeat run (`-run2`) measures the same manuscript twice; it is not a second subject.
_REPEAT = re.compile(r"-run\d+$")


def _records() -> dict[str, dict[str, object]]:
    """Every committed planted-eval record, keyed by manuscript subject."""
    found: dict[str, dict[str, object]] = {}
    for path in sorted((ROOT / "results").glob("planted-eval-*.json")):
        subject = path.stem.removeprefix("planted-eval-")
        if not _REPEAT.search(subject):
            found[subject] = json.loads(path.read_text())
    assert found, "no results/planted-eval-*.json records for the first screen to name"
    return found


def _norm(cell: str) -> str:
    """A cell as text: emphasis, backticks and runs of whitespace are formatting."""
    return re.sub(r"\s+", " ", re.sub(r"[*`_]", "", cell)).strip().lower()


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_divider(line: str) -> bool:
    cells = _cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) is not None for c in cells)


def _tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every markdown pipe table in `text`, as (header cells, data rows)."""
    lines = text.splitlines()
    out: list[tuple[list[str], list[list[str]]]] = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and i + 1 < len(lines) and _is_divider(lines[i + 1]):
            header = _cells(lines[i])
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(_cells(lines[j]))
                j += 1
            out.append((header, rows))
            i = j
        else:
            i += 1
    return out


def _arm_of(label: str) -> str | None:
    """Which arm a row label names: the panel, or one single-agent model.

    The panel row names its MIXTURE (`panel (llama3.1:8b + qwen2:7b)` — D-2), so it
    names both models; being the panel row is therefore checked first, and a baseline
    row must name exactly one model or a reader cannot tell whose pass rate it is.
    """
    text = _norm(label)
    if "panel" in text:
        return PANEL_SYSTEM
    named = [m for m in BASELINE_MODELS if m.lower() in text]
    return baseline_system(named[0]) if len(named) == 1 else None


def _measure_of(header: str) -> str | None:
    """Which measure a column header names. `named token` is tested first: it contains
    the word the tokens column is named for, and reading it as `tokens` would silently
    bind the upper bound to the cost."""
    text = _norm(header)
    if "named" in text:
        return "named token"
    if "assert" in text:
        return "asserted"
    if "wall" in text:
        return "wall"
    if "token" in text:
        return "tokens"
    return None


def _subject_of(header: str, subjects: list[str]) -> str | None:
    """Which manuscript a column header names, by a word of the record's own name."""
    text = _norm(header)
    named = [s for s in subjects if any(w and w in text for w in s.lower().split("-"))]
    assert len(named) <= 1, f"the column header {header!r} names more than one manuscript: {named}"
    return named[0] if named else None


def _eval_table() -> tuple[list[str], dict[str, dict[str, str]]]:
    """The first screen's evaluation table: (header cells, {arm system: {header: cell}}).

    Located by SHAPE — the one table carrying a panel row and a row for each baseline
    model — never by position, so the first screen keeps its freedom to put the table
    wherever it reads best inside the sixty lines.
    """
    wanted = {PANEL_SYSTEM, *(baseline_system(m) for m in BASELINE_MODELS)}
    found = []
    for header, rows in _tables(HEAD):
        arms = {}
        for row in rows:
            arm = _arm_of(row[0])
            if arm is not None:
                arms[arm] = dict(zip(header, row, strict=False))
        if wanted <= set(arms):
            found.append((header, arms, len(rows)))
    assert len(found) == 1, (
        f"README.md lines 1-{len(HEAD_LINES)} carry {len(found)} evaluation tables with a "
        f"row for every arm ({sorted(wanted)}); requirement 10 wants exactly one, with "
        "the assertion-only count and the cost for each named local model"
    )
    header, arms, row_count = found[0]
    assert row_count == len(wanted), (
        f"the evaluation table has {row_count} rows for {len(wanted)} arms — an extra row "
        "is a number nobody bound, and a missing one is a model whose pass rate is unpublished"
    )
    return header, arms


def _count(cell: str, expected: int, n_errors: int, where: str) -> None:
    match = re.search(r"(\d+)\s*(?:/\s*(\d+))?", _norm(cell))
    assert match, f"{where}: {cell!r} carries no count"
    assert int(match.group(1)) == expected, (
        f"{where}: the first screen says {cell!r}; the record says {expected} of {n_errors}"
    )
    if match.group(2) is not None:
        assert int(match.group(2)) == n_errors, (
            f"{where}: the denominator is {match.group(2)}; the record planted {n_errors} errors"
        )


_UNITS = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}


def _wall(cell: str, expected: float, where: str) -> None:
    """The wall-clock cell, in whichever unit it is published, to its own precision."""
    match = re.search(r"([\d,]+(?:\.\d+)?)\s*([a-z]+)?", _norm(cell))
    assert match, f"{where}: {cell!r} carries no wall-clock figure"
    shown = float(match.group(1).replace(",", ""))
    unit = (match.group(2) or "s").strip()
    assert unit in _UNITS, f"{where}: {cell!r} names the unit {unit!r}, not one of {sorted(_UNITS)}"
    decimals = len(match.group(1).split(".")[1]) if "." in match.group(1) else 0
    converted = expected / _UNITS[unit]
    # Half of the cell's own last digit: a published figure must round to the record at
    # whatever precision it chose to publish. A fixed tolerance is what let a measured
    # 3.2x ship as "4x".
    slack = 0.5 * 10.0**-decimals + 1e-9
    assert abs(shown - converted) <= slack, (
        f"{where}: the first screen says {cell!r}; the record's wall_s is {expected} "
        f"({converted:.3f} {unit})"
    )


def _without_code(text: str) -> list[tuple[int, str]]:
    """(1-based line number, line) for every line outside a fenced code block."""
    out: list[tuple[int, str]] = []
    fenced = False
    for number, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            out.append((number, line))
    return out


_QUOTED = re.compile(r'"[^"]*"|“[^”]*”')


def _inside_quotes(line: str, start: int, end: int) -> bool:
    return any(m.start() < start and end <= m.end() for m in _QUOTED.finditer(line))


def _asserted_counts(subject: str, record: dict[str, object]) -> dict[str, int]:
    """{arm system: assertion-credited count} — every arm the code names, or a failure.

    A missing arm is never read as "no win": the question the result sentence answers
    cannot be asked of a record that does not carry all three arms, and a rule whose
    premise is unmeasurable is a rule that cannot fail.
    """
    results = record["results"]
    assert isinstance(results, list)
    counts = {str(r["system"]): len(r["detected"]) for r in results}
    wanted = [PANEL_SYSTEM, *(baseline_system(m) for m in BASELINE_MODELS)]
    missing = [system for system in wanted if system not in counts]
    assert not missing, (
        f"results/planted-eval-{subject}.json carries no arm named {missing} (it has "
        f"{sorted(counts)}); the published result compares the panel with one "
        "single-agent arm per model"
    )
    return {system: counts[system] for system in wanted}


class TestTheFirstScreenFits:
    def test_no_line_of_the_first_screen_is_too_long(self) -> None:
        """Checklist line 1's own command: `head -60 README.md | awk 'length > 160'`."""
        long = [(n + 1, len(line)) for n, line in enumerate(HEAD_LINES) if len(line) > MAX_LINE]
        assert not long, (
            f"lines over {MAX_LINE} characters in the first screen (line, length): {long}"
        )

    def test_the_first_screen_says_this_is_a_demonstration_system(self) -> None:
        assert "demonstration system" in HEAD.lower(), (
            "requirement 10: the first sixty lines must say plainly that this is a "
            "demonstration system"
        )

    def test_the_first_screen_links_the_credibility_map(self) -> None:
        link = re.search(r"\[[^\]]+\]\((?:\./)?CREDIBILITY\.md\)", HEAD)
        assert link, (
            "no markdown link to CREDIBILITY.md in the first sixty lines — requirement 10 "
            "puts the credibility map one click from the first screen"
        )
        assert (ROOT / "CREDIBILITY.md").is_file(), "the first screen links a page that is missing"

    def test_orchestration_is_not_the_headline(self) -> None:
        """The repositioning, in one line: measurement leads, architecture follows."""
        opening = "\n".join(HEAD_LINES[:10]).lower()
        assert "multi-agent orchestration" not in opening, (
            "'multi-agent orchestration' is in the README's first ten lines; requirement "
            "10 says those words are not the headline"
        )


class TestTheEvaluationTableIsBound:
    def test_the_table_carries_one_row_per_arm_and_the_expected_columns(self) -> None:
        header, arms = _eval_table()
        subjects = list(_records())
        groups: dict[str, set[str]] = {}
        for cell in header[1:]:
            subject = _subject_of(cell, subjects)
            measure = _measure_of(cell)
            assert subject is not None and measure is not None, (
                f"the evaluation table's column {cell!r} names "
                f"{'no manuscript' if subject is None else subject} and "
                f"{'no measure' if measure is None else measure}; every column must name "
                f"the manuscript it reports and one of {list(MEASURES)}, so no number on "
                "the first screen is unbindable"
            )
            groups.setdefault(subject, set()).add(measure)
        assert groups, (
            f"none of the evaluation table's columns {header[1:]} names a committed "
            f"manuscript ({subjects}); not one of its numbers could be bound to a record"
        )
        for subject, measures in groups.items():
            missing = [m for m in MEASURES if m not in measures]
            assert not missing, (
                f"{subject}'s columns are missing {missing}: requirement 7 publishes the "
                "assertion-only count as the headline, the named-token upper bound beside "
                "it labelled, and cost as tokens and wall-clock"
            )
        assert set(arms) == {PANEL_SYSTEM, *(baseline_system(m) for m in BASELINE_MODELS)}

    def test_every_cell_of_the_evaluation_table_matches_its_record(self) -> None:
        header, arms = _eval_table()
        records = _records()
        bound = 0
        for cell_header in header[1:]:
            subject = _subject_of(cell_header, list(records))
            measure = _measure_of(cell_header)
            if subject is None or measure is None:
                continue
            record = records[subject]
            results = record["results"]
            assert isinstance(results, list)
            n_errors = int(record["n_errors"])  # type: ignore[call-overload]
            for system, row in arms.items():
                matching = [r for r in results if r["system"] == system]
                assert len(matching) == 1, (
                    f"results/planted-eval-{subject}.json carries "
                    f"{len(matching)} arms named {system!r}: "
                    f"{[r['system'] for r in results]}"
                )
                arm = matching[0]
                assert cell_header in row, f"the {system!r} row has no {cell_header!r} cell"
                cell = row[cell_header]
                where = f"README first screen · {subject} · {system} · {measure}"
                if measure == "asserted":
                    _count(cell, len(arm["detected"]), n_errors, where)
                elif measure == "named token":
                    _count(cell, len(arm["named_token"]), n_errors, where)
                elif measure == "tokens":
                    total = int(arm["total_tokens"])
                    assert f"{total:,}" in cell, (
                        f"{where}: the first screen says {cell!r}; the record's "
                        f"total_tokens is {total:,}"
                    )
                else:
                    _wall(cell, float(arm["wall_s"]), where)
                bound += 1
        assert bound >= len(arms) * len(MEASURES), (
            f"only {bound} cells of the evaluation table were bound; every arm needs all "
            f"of {list(MEASURES)} for at least one manuscript"
        )


# How many rounds came back clean, as the page may honestly say it: the digit, or the
# word a writer reaches for when the answer is none. The count must stand in the same
# clause as `clean` (no `.`, `;` or `,` between them), so it is bound to the word it
# qualifies rather than merely present in the paragraph — and `clean clone`, which is
# how this page describes what an evaluator is handed, is not a clean round.
_ZERO_WORDS = {"not one": 0, "nothing": 0, "none": 0, "never": 0, "zero": 0, "no": 0}
_CLEAN_COUNT = re.compile(
    r"\b(?P<count>not one|nothing|none|never|zero|no|\d+)\b[^.;,]{0,60}?\bclean\b(?!\s+clones?\b)",
    re.IGNORECASE,
)


def _clean_counts(text: str) -> list[int]:
    """Every clean-round count `text` states, in reading order."""
    counts = []
    for match in _CLEAN_COUNT.finditer(text):
        token = match.group("count").lower()
        counts.append(_ZERO_WORDS[token] if token in _ZERO_WORDS else int(token))
    return counts


class TestTheEvaluationLoopLine:
    """Requirement 10's second branch: a sentence about what the review loop found
    either links a committed verbatim excerpt, or says the record is private. This
    repository's gauntlet is private, so the line says so — and its numbers are still
    bound, to `results/evaluation-loop.json`: the rounds completed, the findings they
    filed, and how many of them came back clean — that last one because the record's
    unflattering number is the one a page is tempted to leave out (round 4, report 1,
    finding 3)."""

    def _loop(self) -> dict[str, object]:
        path = ROOT / "results" / "evaluation-loop.json"
        assert path.is_file(), (
            "results/evaluation-loop.json is missing: the first screen's claim about the "
            "review loop has no record behind it. It carries rounds_completed, "
            'findings_by_round, evaluator_sha256 and record: "private".'
        )
        loop: dict[str, object] = json.loads(path.read_text())
        return loop

    def _line(self, sha: str) -> str:
        lines = [line for line in HEAD_LINES if sha[:8] in line]
        assert len(lines) == 1, (
            f"{len(lines)} lines of the first screen carry the evaluator prompt's sha256 "
            f"prefix ({sha[:8]}); requirement 10 wants exactly one line naming the loop, "
            "its rounds, its findings and the pinned prompt"
        )
        return lines[0]

    def test_the_loop_line_names_the_pinned_prompt_and_its_counts(self) -> None:
        loop = self._loop()
        sha = str(loop["evaluator_sha256"])
        line = self._line(sha)
        rounds = re.search(r"(\d+)\s+(?:\w+\s+){0,2}rounds?\b", line, re.IGNORECASE)
        assert rounds, f"the loop line names no round count: {line!r}"
        assert int(rounds.group(1)) == int(loop["rounds_completed"]),  (  # type: ignore[call-overload]
            f"the first screen says {rounds.group(1)} rounds; evaluation-loop.json "
            f"records {loop['rounds_completed']}"
        )
        findings = re.search(r"(\d+)\s+findings\b", line, re.IGNORECASE)
        assert findings, f"the loop line names no findings total: {line!r}"
        by_round = loop["findings_by_round"]
        assert isinstance(by_round, list)
        assert int(findings.group(1)) == sum(int(n) for n in by_round), (
            f"the first screen says {findings.group(1)} findings; evaluation-loop.json "
            f"records {by_round} = {sum(int(n) for n in by_round)}"
        )

    def test_the_loop_line_says_the_record_is_private(self) -> None:
        loop = self._loop()
        assert loop["record"] == "private", (
            "evaluation-loop.json no longer says the record is private; if the gauntlet "
            "has been published, the first screen links the verbatim excerpt instead"
        )
        line = self._line(str(loop["evaluator_sha256"]))
        assert "private" in line.lower(), (
            f"the loop line does not say the record is private: {line!r}. Requirement 10 "
            "allows a sentence about what the loop found only if it links a committed "
            "verbatim excerpt or says the record is private."
        )

    def _block(self, sha: str) -> str:
        """The first-screen paragraph that states the loop.

        The sha line and the lines contiguous with it: a first-screen line is capped at
        160 characters, so the sentence may legitimately wrap and the physical line is
        the wrong unit for a third number. What binds is not presence in the paragraph
        but the count standing in the same clause as the word `clean`.
        """
        index = HEAD_LINES.index(self._line(sha))
        start, end = index, index
        while start > 0 and HEAD_LINES[start - 1].strip():
            start -= 1
        while end + 1 < len(HEAD_LINES) and HEAD_LINES[end + 1].strip():
            end += 1
        return "\n".join(HEAD_LINES[start : end + 1])

    def test_the_loop_line_carries_the_clean_round_count(self) -> None:
        """`rounds_clean`, bound the way `rounds_completed` and `findings_by_round` are.

        Round 4 (report 1, finding 3): the record carries a stated success criterion —
        "a round is clean only when all three reports end in the literal token CLEAN" —
        a cap of 8 rounds, and a met-it-this-often count of 0, and the first screen
        published the round count and the findings total but not that one. It is the
        only self-assessment number in a committed record that the page left out, and
        the only one that makes the sentence harsher rather than softer, which is the
        direction this repository's own standard says must be published.
        """
        loop = self._loop()
        assert "rounds_clean" in loop, (
            "results/evaluation-loop.json no longer carries rounds_clean — the count of "
            "rounds where all three reports ended in CLEAN, which its own _what defines. "
            "The first screen states it, so the record must hold it."
        )
        clean = int(loop["rounds_clean"])  # type: ignore[call-overload]
        completed = loop["rounds_completed"]
        block = self._block(str(loop["evaluator_sha256"]))
        asked = (
            f"the first screen's loop sentence must say how many rounds came back clean "
            f"— the record says {clean} of {completed} — in the same clause as the word "
            f"'clean', e.g. '... and none of the {completed} has come back clean'. The "
            "count may be written as a digit, or as none / no / zero / never when it is 0."
        )
        counts = _clean_counts(block)
        assert counts, (
            f"the loop paragraph states no clean-round count. {asked} It reads: {block!r}"
        )
        wrong = [n for n in counts if n != clean]
        assert not wrong, (
            f"the first screen says {wrong} rounds came back clean; "
            f"results/evaluation-loop.json records rounds_clean = {clean} of {completed}"
        )

    def test_the_clean_count_rule_reads_the_shapes_it_claims_to(self) -> None:
        """A control on the rule above, labelled as one: it is asserted to find a count
        in the real paragraph, so a rule that matched nothing would otherwise read as
        proof that the page was silent about clean rounds."""
        assert _clean_counts(
            "3 blind rounds have filed 78 findings against it, and none of the three has "
            "come back clean (cap 8)."
        ) == [0]
        assert _clean_counts("2 of the 3 rounds came back clean") == [2]
        assert _clean_counts("no round has ever come back clean") == [0]
        # A clean CLONE is what an evaluator is handed, not a clean round; and a sentence
        # that reports only rounds and findings states no clean count at all.
        assert _clean_counts("evaluators handed a clean clone of one commit") == []
        assert _clean_counts(
            "3 blind rounds under a pinned evaluator prompt have filed 78 findings "
            "against it; the reports are this project's private record."
        ) == []


class TestTheResultIsAsTheRecordShowsIt:
    def test_the_word_guards_read_the_shapes_they_claim_to(self) -> None:
        """A control on the regexes themselves, and labelled as one: both are asserted
        ABSENT from the real file below, so a dead regex would otherwise read as proof
        that the prose is honest."""
        assert WIN_WORDS.search("the panel beats the single agent on both manuscripts")
        assert not WIN_WORDS.search("the vector rung wins on every aggregate measure")
        assert LOSS_WORDS.search("the headline is a loss")

    def test_the_result_sentence_follows_the_record(self) -> None:
        wins = []
        for subject, record in _records().items():
            counts = _asserted_counts(subject, record)
            if counts[PANEL_SYSTEM] > max(
                counts[baseline_system(m)] for m in BASELINE_MODELS
            ):
                wins.append(subject)
        if wins:
            # The rule binds a record that shows no win. This one does, on `wins` — so
            # the prose may say so, and this test has nothing left to enforce.
            pytest.skip(f"the panel asserted more than every baseline on {wins}")
        hits = [m.group(0) for m in WIN_WORDS.finditer(HEAD)]
        assert not hits, (
            f"the first screen claims the panel won ({hits}); no committed record shows "
            "the panel asserting more planted errors than every single-agent arm"
        )
        assert LOSS_WORDS.search(HEAD), (
            "no committed record shows the panel winning, and the first screen never says "
            "so: requirement 10 wants the result as the record shows it — a loss or a "
            "null result, never a win"
        )


class TestTheReadmeVoice:
    def test_the_readme_never_speaks_in_the_first_person(self) -> None:
        """"This repository", never "we". The sweep covers the WHOLE README, not the
        first screen: a voice that changes below the fold is the same defect one scroll
        later. Code blocks are not prose, and a quoted citation is someone else's
        sentence — those are the only exemptions.
        """
        pattern = re.compile(r"\b(I|[Mm]y|[Ww]e|[Oo]ur)\b")
        offenders = []
        for number, line in _without_code(README):
            if line.lstrip().startswith(">"):
                continue
            for match in pattern.finditer(line):
                if _inside_quotes(line, match.start(), match.end()):
                    continue
                offenders.append((number, match.group(0), line.strip()[:80]))
        assert not offenders, (
            f"README.md speaks in the first person: {offenders}. The voice is 'this "
            "repository'; a quoted citation is exempt, nothing else."
        )


# ---------------------------------------------------------------------------------
# Figures restated elsewhere on the page (round 5, report 3, finding 1)
#
# The promise this README opens with is made about the PAGE — every number on it is
# read out of a committed record by a test — and an evaluator moved ten published
# figures with the whole suite green. Three of those belong to this file: the first
# screen's exclusion residue, and every index figure the page restates outside the
# bound table (the diagram, and the two-scales paragraph). The grounding yield is the
# fourth, and it carries a second problem: it is published over two runs that are one
# population, so the population is bound here as well as the number.
#
# Each rule is a FUNCTION over the page's text rather than a test over the file, so
# the control beside it can hand the same rule a mutated page and prove it goes red.
# A control that cannot go red is worth nothing — and a binding that cannot find its
# figure binds nothing, which is the defect this whole finding is about, so every
# rule here asserts it located what it claims to check before it checks it.
# ---------------------------------------------------------------------------------

CORPORA = ("ci", "demo")

# The word a figure is written with -> the measure it names. `papers` and `documents`
# are the same count: the manifest's membership list.
_MEASURE_WORDS = {
    "entity": "entities",
    "entities": "entities",
    "edge": "edges",
    "edges": "edges",
    "chunk": "chunks",
    "chunks": "chunks",
    "paper": "documents",
    "papers": "documents",
    "document": "documents",
    "documents": "documents",
}
_FIGURE = re.compile(
    r"(?P<value>\d[\d,]*)[\s-]*"
    r"(?P<word>entities|entity|edges|edge|chunks|chunk|papers|paper|documents|document)\b",
    re.IGNORECASE,
)
# The exclusion residue is a count of chunks that is NOT an index figure; it has its
# own record and its own binding below, and the index rule steps over it.
_DROPPED_CLAIM = re.compile(r"(?P<value>\d[\d,]*)\s+chunks?\s+dropped", re.IGNORECASE)
# A four-digit record value printed on this page is a restatement of that record by
# construction — no unrelated number of that size appears here — so any occurrence the
# figure reader cannot read is a restatement that has quietly stopped being bound.
_READABLE_FLOOR = 1000


def _blocks(text: str) -> list[tuple[int, int, str]]:
    """(first line, last line, text) for every block of the page.

    A fenced block is ONE block: the diagram names the corpus scale in one node and
    the graph size in another, with a blank line between the two halves, and a rule
    that split there would ask the page to repeat the scale inside a picture. Outside
    a fence, a block is a run of non-blank lines — one paragraph, one table, one list.
    """
    lines = text.splitlines()
    blocks: list[tuple[int, int, str]] = []
    current: list[str] = []
    start = 1
    fenced = False
    for number, line in enumerate(lines, start=1):
        if line.strip().startswith("```"):
            if fenced:
                current.append(line)
                blocks.append((start, number, "\n".join(current)))
                current = []
                fenced = False
            else:
                if current:
                    blocks.append((start, number - 1, "\n".join(current)))
                current = [line]
                start = number
                fenced = True
            continue
        if fenced or line.strip():
            if not current:
                start = number
            current.append(line)
        elif current:
            blocks.append((start, number - 1, "\n".join(current)))
            current = []
    if current:
        blocks.append((start, len(lines), "\n".join(current)))
    return blocks


def _block_at(text: str, line: int) -> str:
    for start, end, block in _blocks(text):
        if start <= line <= end:
            return block
    raise AssertionError(f"README.md line {line} sits in no block")


def _line_of(text: str, offset: int) -> int:
    return text[:offset].count("\n") + 1


def _corpus_figures() -> dict[str, dict[str, int]]:
    """{corpus: {measure: value}} — the index figures, read from the records.

    `documents` is the manifest's own membership list; the other three come from
    `results/build-stats-<corpus>.json`. Nothing here is computed by this test.
    """
    figures: dict[str, dict[str, int]] = {}
    for corpus in CORPORA:
        stats_path = ROOT / "results" / f"build-stats-{corpus}.json"
        manifest_path = ROOT / "corpus" / f"{corpus}.manifest.json"
        assert stats_path.is_file(), (
            f"{stats_path.relative_to(ROOT)} is missing: the index figures this page "
            "restates have no record to be read out of"
        )
        assert manifest_path.is_file(), (
            f"{manifest_path.relative_to(ROOT)} is missing: the document count this page "
            "restates has no record to be read out of"
        )
        stats = json.loads(stats_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        figures[corpus] = {
            "documents": len(manifest["docs"]),
            "chunks": int(stats["chunks"]),
            "entities": int(stats["nodes"]),
            "edges": int(stats["edges"]),
        }
    return figures


def _bind_dropped_chunks(text: str) -> int:
    """The exclusion residue, bound to the planted-eval record of the held-in twin.

    The page publishes one figure for every arm — "26 chunks dropped on each" — so the
    record must show every arm dropping the same number, and the manuscript the
    sentence is about is read from `twin_in_corpus` rather than from the sentence: the
    paragraph names both manuscripts, and the figure belongs to exactly one of them.
    """
    records = _records()
    held = sorted(s for s, r in records.items() if r.get("twin_in_corpus"))
    assert len(held) == 1, (
        f"{len(held)} committed planted-eval records say their twin is a corpus member "
        f"({held}); the first screen's sentence is about the one manuscript that is not "
        "fully held out, and its figure is only bindable while there is exactly one"
    )
    subject = held[0]
    results = records[subject]["results"]
    assert isinstance(results, list)
    per_arm = {str(arm["system"]): int(arm["dropped_chunks"]) for arm in results}
    assert per_arm, (
        f"results/planted-eval-{subject}.json carries no arm, so there is no "
        "dropped_chunks the page could be quoting"
    )
    values = sorted(set(per_arm.values()))
    claims = list(_DROPPED_CLAIM.finditer(text))
    assert claims, (
        "README.md states no chunks-dropped figure this binding can read. The exclusion "
        f"residue is published as '(N chunks dropped on each)' — the record says "
        f"{values} — and written any other way it is bound by nothing, which is the "
        "defect this rule exists to close."
    )
    for match in claims:
        line = _line_of(text, match.start())
        block = _block_at(text, line)
        assert subject in block, (
            f"README.md line {line} states {match.group(0)!r} in a paragraph that never "
            f"names {subject}, the manuscript whose twin is a corpus member; the figure "
            "is per-manuscript, so the paragraph must say whose it is"
        )
        assert len(values) == 1, (
            f"README.md line {line} publishes one chunks-dropped figure for every arm, "
            f"but results/planted-eval-{subject}.json records {per_arm}: publish it per "
            "arm, or as a range — 'on each' is a claim the record no longer supports"
        )
        shown = int(match.group("value").replace(",", ""))
        assert shown == values[0], (
            f"README.md line {line} says {match.group(0)!r}; every arm in "
            f"results/planted-eval-{subject}.json records dropped_chunks = {values[0]}"
        )
    return len(claims)


def _bind_index_figures(text: str) -> list[tuple[int, str, str, int]]:
    """Every restated index figure on the page, bound to the record of its own scale.

    Two corpora are published here, so a figure is only bound once its SCALE is known.
    The rule the page already follows: a document count names the scale (15 papers, 68
    papers — each manifest's own length), and every entity, edge and chunk count after
    it in the same block belongs to that corpus. A figure with no scale named before it
    fails, because a bare '7,697 entities' is a number a reader cannot check.
    """
    figures = _corpus_figures()
    bound: list[tuple[int, str, str, int]] = []
    for start, _end, block in _blocks(text):
        skip = [m.span() for m in _DROPPED_CLAIM.finditer(block)]
        scale: str | None = None
        for match in _FIGURE.finditer(block):
            if any(a <= match.start() and match.end() <= b for a, b in skip):
                continue
            measure = _MEASURE_WORDS[match.group("word").lower()]
            value = int(match.group("value").replace(",", ""))
            line = start + block[: match.start()].count("\n")
            where = f"README.md line {line}: {match.group(0).strip()!r}"
            if measure == "documents":
                named = sorted(c for c, f in figures.items() if f["documents"] == value)
                assert len(named) == 1, (
                    f"{where} names a corpus of {value} documents; the committed "
                    f"manifests hold "
                    f"{ {c: f['documents'] for c, f in sorted(figures.items())} }. A "
                    "document count on this page is what tells a reader which scale the "
                    "figures beside it belong to, so it must be one of them."
                )
                scale = named[0]
            else:
                assert scale is not None, (
                    f"{where} restates an index figure before any document count has "
                    "named the scale it belongs to. Two corpora are published here "
                    f"({ {c: f['documents'] for c, f in sorted(figures.items())} } "
                    "documents), so name the scale first — 'the 68-paper open-access "
                    f"corpus … {value:,} {measure}' — or drop the restatement."
                )
                expected = figures[scale][measure]
                assert value == expected, (
                    f"{where} is read against results/build-stats-{scale}.json, which "
                    f"records {expected:,} {measure}. (The scale was set by the document "
                    f"count before it: {figures[scale]['documents']} documents.)"
                )
            bound.append((line, scale, measure, value))
    assert bound, (
        "README.md restates no index figure this binding can read. Today the diagram "
        "and the two-scales paragraph both carry them; if a rewrite has removed every "
        "restatement, this rule has nothing left to bind and goes with it — but a "
        "figure that is still on the page and no longer readable here is the defect."
    )
    return bound


def _bind_restated_values_are_readable(text: str) -> int:
    """Every committed index value printed on the page must be READ by the rule above.

    The rule above can only bind a figure it can parse, so a rewrite that keeps the
    number and drops the word ('entities: 7,697') would silently bind nothing and pass
    — the exact shape of the finding this file is closing. So each record value of
    four digits or more is searched for as a bare number, and every occurrence must sit
    inside a figure the reader located. Removing a restatement stays free; making one
    unreadable does not.
    """
    figures = _corpus_figures()
    readable = [m.span("value") for m in _FIGURE.finditer(text)]
    readable += [m.span("value") for m in _DROPPED_CLAIM.finditer(text)]
    checked = 0
    for corpus, measures in sorted(figures.items()):
        for measure, value in sorted(measures.items()):
            if value < _READABLE_FLOOR:
                continue
            forms = {f"{value:,}", str(value)}
            for form in sorted(forms):
                pattern = re.compile(rf"(?<![\d,.]){re.escape(form)}(?![\d,.])")
                for match in pattern.finditer(text):
                    checked += 1
                    covered = any(a <= match.start() and match.end() <= b for a, b in readable)
                    assert covered, (
                        f"README.md line {_line_of(text, match.start())} carries {form}, "
                        f"which is results/build-stats-{corpus}.json's own {measure} "
                        "count, in a form this binding cannot read. A restated record "
                        f"number must be written so it can be bound — '{form} {measure}' "
                        "after the scale is named — or removed from the page."
                    )
    assert checked, (
        "no committed index value of four digits or more appears on this page, so this "
        "rule read nothing; while build-stats-demo.json holds "
        f"{ {m: v for m, v in sorted(figures['demo'].items()) if v >= _READABLE_FLOOR} }, "
        "that means the reader, not the page, has stopped working"
    )
    return checked


# The one field a rerun of this panel may legitimately move. At temperature 0 the panel
# is deterministic — RESULTS.md says the two committed runs differ in `wall_s` alone —
# so a second record that differs in nothing else is a REPRODUCTION of the first, not a
# second sample, and a total taken across both counts one population twice. Reading that
# from the records rather than asserting it means a genuinely independent second run
# would relax the rule on its own.
PANEL_RUN_VARYING = ("wall_s",)
_YIELD_PER_RUN = re.compile(
    r"(?P<num>\d[\d,]*)(?:\s+of\s+(?P<den>\d[\d,]*))?(?:\s+verdicts?)?"
    r"\s+in each\b[^.;]{0,60}?\bruns?\b",
    re.IGNORECASE,
)
_YIELD_TOTAL = re.compile(
    r"(?P<num>\d[\d,]*)\s+of\s+(?P<den>\d[\d,]*)\s+verdicts\b", re.IGNORECASE
)
# Either form says the same thing: the second run repeats the first, so the combined
# figure is one population counted twice. The prose may say it in its own words; these
# are the shapes this rule reads.
_REPRODUCTION_CAVEATS = (
    re.compile(r"\breproduction\b[^.]{0,90}\bnot an independent sample\b", re.IGNORECASE),
    re.compile(r"\bcounts? the same \d[\d,]* verdicts twice\b", re.IGNORECASE),
)


def _panel_runs() -> dict[str, dict[str, object]]:
    found = {
        path.name: json.loads(path.read_text())
        for path in sorted((ROOT / "results").glob("panel-review-*.json"))
    }
    assert found, (
        "no results/panel-review-*.json records: the grounding yield this page "
        "publishes has nothing to be read out of"
    )
    return found


def _without_run_varying(value: object) -> object:
    if isinstance(value, dict):
        return {k: _without_run_varying(v) for k, v in value.items() if k not in PANEL_RUN_VARYING}
    if isinstance(value, list):
        return [_without_run_varying(v) for v in value]
    return value


def _one_population(runs: dict[str, dict[str, object]]) -> bool:
    """Are these records the same run, published twice?"""
    stripped = [_without_run_varying(record) for record in runs.values()]
    return len(stripped) > 1 and all(record == stripped[0] for record in stripped[1:])


def _bind_grounding_yield(text: str) -> int:
    """The grounding yield, bound PER RUN, with the doubled population named as one.

    A verdict is grounded when it carries evidence that survived the citation check —
    the same definition `test_artifact_conformance.py` computes from — so the per-run
    figure is that count against that run's verdicts. The combined figure is the one
    that needs care: the two committed runs are a reproduction (they differ in
    `wall_s` alone, which this rule recomputes), so 12 of 40 is 6 of 20 counted twice
    and may only be published saying so.
    """
    runs = _panel_runs()
    per_run: dict[str, tuple[int, int]] = {}
    for name, record in runs.items():
        verdicts = record["verdicts"]
        assert isinstance(verdicts, list), f"results/{name} carries no verdicts list"
        per_run[name] = (sum(1 for v in verdicts if v.get("evidence")), len(verdicts))
    grounded = sorted({g for g, _ in per_run.values()})
    totals = sorted({t for _, t in per_run.values()})
    bound = 0

    claims = list(_YIELD_PER_RUN.finditer(text))
    assert claims, (
        "README.md publishes no per-run grounding yield. The records say "
        f"{ {name: f'{g} of {t}' for name, (g, t) in sorted(per_run.items())} }, and "
        "the per-run figure is the one an n of two identical runs supports: write it as "
        f"'{grounded[0]} of {totals[0]} verdicts in each of the {len(runs)} committed "
        "runs'."
    )
    for match in claims:
        line = _line_of(text, match.start())
        assert len(grounded) == 1, (
            f"README.md line {line} publishes one grounding yield for every run, but the "
            f"records disagree: { {name: g for name, (g, _t) in sorted(per_run.items())} }. "
            "Publish it per run, or the page states a figure no record holds."
        )
        shown = int(match.group("num").replace(",", ""))
        assert shown == grounded[0], (
            f"README.md line {line} says {match.group(0).strip()!r}; every committed "
            f"panel run grounds {grounded[0]} of its verdicts "
            f"({ {name: f'{g} of {t}' for name, (g, t) in sorted(per_run.items())} })"
        )
        if match.group("den") is not None:
            denominator = int(match.group("den").replace(",", ""))
            assert len(totals) == 1 and denominator == totals[0], (
                f"README.md line {line} says {match.group(0).strip()!r}; a committed run "
                f"carries {totals if len(totals) > 1 else totals[0]} verdicts"
            )
        bound += 1

    per_run_spans = [m.span() for m in claims]
    for match in _YIELD_TOTAL.finditer(text):
        if any(a <= match.start() and match.end() <= b for a, b in per_run_spans):
            continue
        line = _line_of(text, match.start())
        shown = int(match.group("num").replace(",", ""))
        denominator = int(match.group("den").replace(",", ""))
        expected_num = sum(g for g, _ in per_run.values())
        expected_den = sum(t for _, t in per_run.values())
        assert (shown, denominator) == (expected_num, expected_den), (
            f"README.md line {line} says {match.group(0).strip()!r}; summed over the "
            f"committed runs the records hold {expected_num} of {expected_den} "
            f"({ {name: f'{g} of {t}' for name, (g, t) in sorted(per_run.items())} })"
        )
        if _one_population(runs):
            block = _block_at(text, line)
            assert any(pattern.search(block) for pattern in _REPRODUCTION_CAVEATS), (
                f"README.md line {line} publishes {match.group(0).strip()!r} as though it "
                f"were an n of {expected_den}. The {len(runs)} committed panel records are "
                f"identical apart from {list(PANEL_RUN_VARYING)}, so the second run is a "
                "reproduction of the first and this figure counts one population twice. "
                "Lead with the per-run figure and name the doubling in the same "
                f"paragraph, e.g.: 'on the rebuilt index it is **{grounded[0]} of "
                f"{totals[0]} verdicts in each of the {len(runs)} committed runs** — the "
                "second run is a reproduction of the first, not an independent sample, "
                f"so the combined {expected_num} of {expected_den} verdicts counts the "
                f"same {totals[0]} twice.'"
            )
        bound += 1
    return bound


# A wall-clock timing printed beside a command: `~3s`, `~1 min`, `~12h`. No record
# holds these — nothing measures the reader's machine — so they are the one place the
# page's universal binding promise cannot be made true, and it has to say so.
_TIMING = re.compile(
    r"~\s*\d+(?:\.\d+)?\s*(?:s\b|secs?\b|seconds?\b|m\b|mins?\b|minutes?\b|h\b|hours?\b)",
    re.IGNORECASE,
)
_PROMISE = re.compile(r"\bevery number\b", re.IGNORECASE)
# The promise carries its own exception ...
_SCOPED_PROMISE = re.compile(
    r"\b(?:except|apart from|other than|excluding)\b[^.]{0,120}"
    r"\b(?:timings?|wall-clock|wall clock|runtimes?|seconds?|minutes?)\b",
    re.IGNORECASE,
)
# ... or the timings carry their own, where they are printed.
_TIMING_CAVEATS = (
    re.compile(r"\bmeasured on one laptop\b", re.IGNORECASE),
    re.compile(r"\bon (?:one|a|an)\b[^.\n]{0,24}\blaptop\b", re.IGNORECASE),
    re.compile(
        r"\bfrom no record\b|\bno committed record\b|\bnot read (?:out )?of (?:a|any) record\b",
        re.IGNORECASE,
    ),
)


def _bind_the_promise_is_scoped(text: str) -> int:
    """The one figure on this page that cannot be bound must be scoped out of the claim.

    D-14: the promise is made true rather than narrowed — except here, where there is
    no record to make it true against. So either the promise names the exception, or
    every block that prints a timing says where the timing came from.
    """
    blocks = _blocks(text)
    promises = [(start, block) for start, _end, block in blocks if _PROMISE.search(block)]
    assert len(promises) == 1, (
        f"{len(promises)} blocks of README.md carry the phrase 'every number'; this rule "
        "binds the page's one binding promise, and it can only do that while there is "
        "exactly one"
    )
    promise_start, promise = promises[0]
    timed = [
        (start, block, [m.group(0) for m in _TIMING.finditer(block)])
        for start, _end, block in blocks
        if _TIMING.search(block)
    ]
    assert timed, (
        "README.md prints no wall-clock timing this rule can find. It printed `~3s`, "
        "`~3s` and `~1 min` beside the quickstart commands, and this rule exists to keep "
        "the binding promise off them; if the timings are gone the promise covers "
        "nothing unbindable, and this rule goes with them."
    )
    if _SCOPED_PROMISE.search(promise):
        return len(timed)
    unscoped = [
        (start, marks)
        for start, block, marks in timed
        if not any(pattern.search(block) for pattern in _TIMING_CAVEATS)
    ]
    assert not unscoped, (
        f"README.md line {promise_start} promises every number on this page is read out "
        "of a committed record by a test, and these wall-clock timings come from no "
        f"record: { {f'line {start}': marks for start, marks in unscoped} }. Scope the "
        "promise where it stands, e.g.: 'Every number on this page is read out of a "
        "committed record under [results/](results/) by a test in this repository — "
        "including the numbers that look bad — apart from the wall-clock timings beside "
        "the commands below, which are measured on one laptop and come from no record.' "
        "Or leave the promise and caveat each timing where it is printed, e.g. '~3s, "
        "measured on one laptop'."
    )
    return len(timed)


def _mutated(old: str, new: str) -> str:
    """The page with one published string changed — the evaluator's own experiment.

    The mutation is asserted to have applied. A control that silently changed nothing
    proves only that the binding still passes on the file it already passed on, which
    is the shape of green this repository keeps finding under its own claims.
    """
    assert old in README_PATH.read_text(), (
        f"README.md no longer carries {old!r}, so this control mutates nothing; "
        "re-derive it from the page before trusting the green beside it"
    )
    changed = README.replace(old, new, 1)
    assert changed != README, f"the control mutation {old!r} -> {new!r} changed nothing"
    return changed


class TestTheExclusionResidueIsBound:
    """The first screen's "26 chunks dropped on each", against the record that
    measured it (round 5, report 3, finding 1: it moved to 36 with 594 tests green)."""

    def test_the_dropped_chunks_figure_matches_the_record(self) -> None:
        assert _bind_dropped_chunks(README) >= 1

    def test_a_moved_dropped_chunks_figure_is_caught(self) -> None:
        with pytest.raises(AssertionError, match="dropped_chunks"):
            _bind_dropped_chunks(_mutated("26 chunks dropped", "36 chunks dropped"))

    def test_a_dropped_chunks_sentence_with_no_figure_is_caught(self) -> None:
        """Non-vacuity: a rewrite that drops the number fails loudly rather than
        leaving this rule quietly binding nothing."""
        with pytest.raises(AssertionError, match="states no chunks-dropped figure"):
            _bind_dropped_chunks(
                _mutated("(26 chunks dropped on each)", "(chunks are dropped on each)")
            )


class TestTheRestatedIndexFiguresAreBound:
    """The diagram and the two-scales paragraph restate the index figures the table in
    `results/RESULTS.md` binds. Restated, they were bound by nothing: an evaluator
    moved 7,697 to 7,997, 63,130 to 66,130 and 68 papers to 88 with the suite green."""

    def test_every_restated_index_figure_matches_its_record(self) -> None:
        assert _bind_index_figures(README)

    def test_the_figures_inside_the_diagram_are_bound_too(self) -> None:
        """The diagram is a fenced block, and a block walk that stopped at the fence
        would skip it silently — so this asserts at least one bound figure stands
        inside one."""
        fenced = [
            (start, end)
            for start, end, block in _blocks(README)
            if block.lstrip().startswith("```")
        ]
        inside = [f for f in _bind_index_figures(README) if any(s <= f[0] <= e for s, e in fenced)]
        assert inside, (
            "no index figure inside a fenced block was bound; the diagram carries "
            "'7,697 entities · 63,130 edges' today, and a rule that cannot see inside "
            "the fence binds nothing there"
        )

    def test_a_moved_figure_in_the_diagram_is_caught(self) -> None:
        with pytest.raises(AssertionError, match=r"build-stats-demo\.json"):
            _bind_index_figures(
                _mutated("7,697 entities · 63,130 edges", "7,997 entities · 66,130 edges")
            )

    def test_a_moved_document_count_in_the_diagram_is_caught(self) -> None:
        with pytest.raises(AssertionError, match="names a corpus of 88 documents"):
            _bind_index_figures(_mutated("68 papers, demo scale", "88 papers, demo scale"))

    def test_a_moved_figure_in_the_prose_is_caught(self) -> None:
        with pytest.raises(AssertionError, match=r"build-stats-demo\.json"):
            _bind_index_figures(
                _mutated(
                    "1,077 chunks, 7,697 entities, 63,130 edges",
                    "1,077 chunks, 7,997 entities, 66,130 edges",
                )
            )

    def test_a_figure_from_the_other_corpus_is_caught(self) -> None:
        """Two scales are published here, and both are committed numbers — so a figure
        is only bound once the scale it belongs to is known. This swaps the demo
        diagram's figures for the CI corpus's real ones."""
        with pytest.raises(AssertionError, match=r"build-stats-demo\.json"):
            _bind_index_figures(
                _mutated("7,697 entities · 63,130 edges", "1,759 entities · 13,516 edges")
            )

    def test_a_figure_with_no_scale_named_before_it_is_caught(self) -> None:
        with pytest.raises(AssertionError, match="named the scale"):
            _bind_index_figures(
                _mutated(
                    "Open-access corpus<br/>68 papers, demo scale",
                    "Open-access corpus<br/>demo scale",
                )
            )


class TestEveryRestatedRecordValueIsReadable:
    """The non-vacuity rule for the one above: a restatement that keeps the number and
    loses the word would bind nothing and pass, which is this finding's whole shape."""

    def test_every_committed_index_value_on_the_page_was_read(self) -> None:
        assert _bind_restated_values_are_readable(README)

    def test_a_restatement_this_binding_cannot_read_is_caught(self) -> None:
        with pytest.raises(AssertionError, match="in a form this binding cannot read"):
            _bind_restated_values_are_readable(
                _mutated("7,697 entities · 63,130 edges", "entities: 7,697 · edges: 63,130")
            )


class TestTheGroundingYieldIsBound:
    """The yield of the grounding check, per run — and the doubled n named as one.

    Round 5 moved "12 of 40 verdicts" to "19 of 40" with the suite green (report 3,
    finding 1), and finding 12 is the other half: the two committed runs are one
    population, so 40 is 20 counted twice and the page has to say so.
    """

    def test_the_yield_is_published_per_run_with_the_doubling_named(self) -> None:
        assert _bind_grounding_yield(README)

    def test_a_moved_per_run_yield_is_caught(self) -> None:
        with pytest.raises(AssertionError, match="every committed panel run grounds"):
            _bind_grounding_yield(
                _mutated("6 of 20 verdicts in each", "9 of 20 verdicts in each")
            )

    def test_a_page_with_no_per_run_yield_is_caught(self) -> None:
        with pytest.raises(AssertionError, match="publishes no per-run grounding yield"):
            _bind_grounding_yield(
                _mutated("in each of the two committed", "across the two committed")
            )

    def test_a_moved_combined_yield_is_caught(self) -> None:
        with pytest.raises(AssertionError, match="summed over the"):
            _bind_grounding_yield(
                _mutated("12 of 40 verdicts counts", "19 of 40 verdicts counts")
            )

    def test_the_prescribed_sentence_satisfies_this_rule(self) -> None:
        """The instruction the failure carries is satisfiable, proven rather than
        promised: a red that cannot be answered is worse than no red at all."""
        # The page now carries the per-run lead, so this control proves the OTHER
        # accepted shape also satisfies the rule: the combined figure kept, with the
        # reproduction named in the same sentence. A control pinned to the wording the
        # page happens to use would only re-assert the test above.
        assert _bind_grounding_yield(
            _mutated(
                "it is **6 of 20 verdicts in each of the two committed\n"
                "runs** — the second run is a reproduction of the first, not an "
                "independent sample, so the combined\n12 of 40 verdicts counts the "
                "same 20 twice.",
                "it is **12 of 40 verdicts**, 6 in each run — the second run is a\n"
                "reproduction of the first, not an independent sample, so it counts "
                "the same 20 verdicts twice.",
            )
        )


class TestTheUnbindableTimingsAreScopedOut:
    """The quickstart's `~3s` / `~1 min` come from no record, and no test can bind
    them. D-14: that one place is scoped out of the promise rather than quietly
    covered by it."""

    def test_the_binding_promise_does_not_cover_the_wall_clock_timings(self) -> None:
        assert _bind_the_promise_is_scoped(README)

    def test_the_prescribed_scoping_sentence_satisfies_this_rule(self) -> None:
        # The page now carries the scoped promise, so this control proves the other
        # accepted road: the promise left universal and each timing caveated where it
        # is printed.
        assert _bind_the_promise_is_scoped(
            _mutated(
                " — apart from the wall-clock timings beside the commands below, which "
                "are measured on one laptop and come from no record.",
                ".",
            ).replace("# ~3s: rebuilds", "# ~3s (measured on one laptop): rebuilds")
        )

    def test_a_promise_that_still_covers_the_timings_is_caught(self) -> None:
        """The control on the rule itself: with the scoping sentence in place it
        passes (above), and with the scope taken back out it goes red."""
        unscoped = _mutated(
            " — apart from the wall-clock timings beside the commands below, which "
            "are measured on one laptop and come from no record.",
            ".",
        )
        with pytest.raises(AssertionError, match="come from no"):
            _bind_the_promise_is_scoped(unscoped)
