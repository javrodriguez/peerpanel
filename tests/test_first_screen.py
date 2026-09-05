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
