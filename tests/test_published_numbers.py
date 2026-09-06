"""Every number in RESULTS.md and README.md is bound to the artifact it describes.

This file exists because a whole latency column went stale with 268 tests green.
The prose and the JSON were each correct at different moments, one commit
regenerated the artifact and rewrote the surrounding sentences, and nothing in
the repo noticed that one column still described the superseded run — which
inflated a published "22x slower" headline that a conclusion rested on.

That is the D11 shape (a check that cannot fail when the thing it guards stops
working) on the repo's most load-bearing surface: the numbers a reader is asked
to trust. `test_prose_invariants.py` checks that the files EXIST and that no
denied phrase appears; nothing compared a value to its source.

So: these tests parse the published tables and recompute every figure from
results/*.json. A number that drifts from its artifact fails here, loudly, with
both values named.

Round 3 then found this file wearing the same defect it was written to catch:
three guards whose regexes matched NOTHING in either file, so the numbers they
covered had drifted with the guards green (`gauntlet/round-3/report-2.md`
Finding 6). Two rules answer that, and they apply to everything added here:

1. **Every binding regex asserts it matched at least one real string**, in the
   real file — never a literal in the test. A guard over zero strings is a guard
   that cannot fail.
2. **A cell is located by ROW LABEL and COLUMN HEADER, structurally**, never by
   "the string appears somewhere in the document". `_tables()` parses the
   markdown; a table that loses a row, renames a label or drops a column fails on
   the shape, before any number is read.

A published figure is checked against the artifact AT THE PRECISION IT IS
PUBLISHED TO: "3x" must round to the artifact's ratio, "3.2x" must round to it
one digit finer. A fixed tolerance is what let a published 4x stand against a
measured 3.2x.

The planted-evaluation bindings live in `test_planted_published_numbers.py` —
they moved out when two owners needed to edit this file at once.

Round 6 added two more pages to this file's reach, for the same reason it exists.
`LIMITATIONS.md:7` promises that a figure that drifts fails the suite, and three of
its own figures were pinned in test DOCSTRINGS only (report 1, finding 3) — a claim
about the repository's verification discipline that the repository did not keep, on
the page the README calls the one most worth reading carefully. And `results/
RESULTS.md`'s model-layer drift figure had no artifact, no test and no offline route
(finding 4). Both are bound below: the first to the graph rebuilt from
`fixtures/extraction/ci`, the second as far as it CAN be bound offline — its
population to the committed reports, its numerator to an attribution a reader can
follow, because no committed byte can reproduce a second cold model generation.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from functools import cache
from pathlib import Path
from typing import Any, ClassVar

import pytest

from peerpanel.agents.schemas import RUBRIC_DIMENSIONS, VERDICTS
from peerpanel.corpus.models import CorpusManifest
from peerpanel.graph.run_graph import build_run_graph
from peerpanel.manuscripts.store import read_manuscript

ROOT = Path(__file__).resolve().parents[1]
RESULTS = (ROOT / "results" / "RESULTS.md").read_text()
README = (ROOT / "README.md").read_text()
LIMITATIONS = (ROOT / "LIMITATIONS.md").read_text()

# Globbed, never hardcoded: these files were renamed once already, and a hardcoded name
# turns into an unconditional FileNotFoundError that reads like a broken test rather than
# an unbound number.
PANEL_RECORDS = sorted((ROOT / "results").glob("panel-review-*.json"))

ROW_LABELS = {
    "BM25": "bm25",
    "vector": "vector",
    "RRF hybrid": "rrf-hybrid",
    "GraphRAG local": "graphrag-local",
    "GraphRAG global": "graphrag-global",
}

# A per-case row may name more than one rung ("BM25 · RRF"), so the label is matched by
# the words that distinguish a rung rather than by its full display name.
RUNG_ALIASES = {
    "bm25": ("bm25",),
    "vector": ("vector",),
    "rrf-hybrid": ("rrf",),
    "graphrag-local": ("graphrag", "local"),
    "graphrag-global": ("graphrag", "global"),
}


# --------------------------------------------------------------------------------------
# Structural markdown-table parsing. Everything below locates a cell by row label and
# column header; nothing searches a document for a value.
# --------------------------------------------------------------------------------------


def _norm(cell: str) -> str:
    """A cell as text: emphasis, backticks and runs of whitespace are formatting."""
    return re.sub(r"\s+", " ", re.sub(r"[*`_]", "", cell)).strip().lower()


def _cells(line: str) -> list[str]:
    stripped = line.strip()
    return [c.strip() for c in stripped.strip("|").split("|")]


def _is_divider(line: str) -> bool:
    cells = _cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) is not None for c in cells)


def _tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Every markdown pipe table in `text`, as (header cells, data rows).

    Cells are returned raw — normalisation is the caller's, because a label and a
    number want different treatment. Empty header cells are kept: the index table's
    first column has no heading, and dropping it would shift every column by one.
    """
    lines = text.splitlines()
    tables: list[tuple[list[str], list[list[str]]]] = []
    i = 0
    while i < len(lines):
        if (
            lines[i].strip().startswith("|")
            and i + 1 < len(lines)
            and _is_divider(lines[i + 1])
        ):
            header = _cells(lines[i])
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(_cells(lines[j]))
                j += 1
            tables.append((header, rows))
            i = j
        else:
            i += 1
    return tables


def _numbers(cell: str) -> list[float]:
    """Every number in a cell, commas and emphasis stripped, in order."""
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", _norm(cell))]


def _slack(shown: str) -> float:
    """Half of the last digit the claim published — the honest rounding window.

    "3x" must round to the artifact's ratio; "3.2x" must round to it one digit
    finer. A fixed tolerance is how a published 4x survived a measured 3.2x.
    """
    decimals = len(shown.split(".")[1]) if "." in shown else 0
    return 0.5 * 10.0**-decimals + 1e-9


def _ablation(corpus: str = "demo") -> dict[str, list[dict[str, object]]]:
    report = json.loads((ROOT / "results" / f"ablation-{corpus}.json").read_text())
    by_rung: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in report["results"]:
        by_rung[str(row["rung"])].append(row)
    return by_rung


def _cell(value: str) -> float:
    return float(re.sub(r"[^0-9.]", "", value))


def _published_demo_rows() -> dict[str, list[str]]:
    """The main demo table in RESULTS.md, keyed by rung."""
    rows: dict[str, list[str]] = {}
    for label, rung in ROW_LABELS.items():
        # labels may be bolded when a rung leads its column
        match = re.search(rf"^\| \**{re.escape(label)}\** \|(.+)$", RESULTS, re.MULTILINE)
        assert match, f"no published row for {label}"
        rows[rung] = [c.strip() for c in match.group(1).split("|") if c.strip()]
    return rows


class TestAblationTableMatchesItsArtifact:
    """Each published cell against the JSON RESULTS.md calls the record."""

    def test_found_counts(self) -> None:
        data, published = _ablation(), _published_demo_rows()
        for rung, cells in published.items():
            rows = data[rung]
            k = int(json.loads((ROOT / "results" / "ablation-demo.json").read_text())["k"])
            at_k = sum(1 for r in rows for _, rank in r["hits"] if rank is not None and rank <= k)
            deep = sum(1 for r in rows for _, rank in r["hits"] if rank is not None)
            assert _cell(cells[0].split("/")[0]) == at_k, f"{rung} found@k"
            assert _cell(cells[1].split("/")[0]) == deep, f"{rung} found@deep"

    def test_recall_ceiling_and_ndcg(self) -> None:
        data, published = _ablation(), _published_demo_rows()
        for rung, cells in published.items():
            rows = data[rung]
            recall = sum(float(r["recall_at_k"]) for r in rows) / len(rows)  # type: ignore[arg-type]
            ceiling = sum(float(r["recall_ceiling_at_k"]) for r in rows) / len(rows)  # type: ignore[arg-type]
            ndcg = sum(float(r["ndcg_at_k"]) for r in rows) / len(rows)  # type: ignore[arg-type]
            assert _cell(cells[2]) == pytest.approx(recall, abs=0.001), f"{rung} recall"
            assert _cell(cells[3]) == pytest.approx(ceiling, abs=0.001), f"{rung} ceiling"
            # Macro-average, matching every other rate in the table. A
            # ratio-of-means reweights toward the case with the larger
            # denominator and inverted the apparent winner when it was used.
            shares = [
                float(r["recall_at_k"]) / float(r["recall_ceiling_at_k"])  # type: ignore[arg-type]
                for r in rows
            ]
            macro = sum(shares) / len(shares) * 100
            assert _cell(cells[4]) == pytest.approx(macro, abs=1), f"{rung} %"
            assert _cell(cells[5]) == pytest.approx(ndcg, abs=0.001), f"{rung} ndcg"

    def test_latency_column_is_not_stale(self) -> None:
        """The column that went stale. Bound to within a millisecond of the run."""
        data, published = _ablation(), _published_demo_rows()
        for rung, cells in published.items():
            rows = data[rung]
            latency = sum(float(r["latency_ms"]) for r in rows) / len(rows)  # type: ignore[arg-type]
            assert _cell(cells[6]) == pytest.approx(latency, abs=1.0), (
                f"{rung}: RESULTS.md says {cells[6]}, the artifact says {latency:.1f} ms"
            )

    def test_the_speed_ratio_claim(self) -> None:
        """Any 'Nx <something>' latency claim must match the artifact it summarises.

        Round-3 F6a: the old regex demanded the words "slower" or "faster", which
        appear after no multiplier in either file, so it bound ZERO strings while both
        live claims — "GraphRAG local runs 4x that" and "at 4x BM25's latency" — sat
        unguarded against a measured 3.2x.

        Every `Nx` claim in these two files compares the slowest graph rung with the
        lexical baseline; that is the only latency ratio either document draws. A claim
        about anything else must not be written in this shape.
        """
        data = _ablation()

        def mean(rung: str) -> float:
            rows = data[rung]
            return sum(float(r["latency_ms"]) for r in rows) / len(rows)  # type: ignore[arg-type]

        real = mean("graphrag-local") / mean("bm25")
        pattern = re.compile(
            r"(\d+(?:\.\d+)?)\s*[x\u00d7]\s*(?:slower|faster|that|BM25|the (?:cost|latency))",
            re.IGNORECASE,
        )
        seen = 0
        for text, where in ((RESULTS, "results/RESULTS.md"), (README, "README.md")):
            for match in pattern.finditer(text):
                seen += 1
                claimed = match.group(1)
                assert float(claimed) == pytest.approx(real, abs=_slack(claimed)), (
                    f"{where} claims {match.group(0)!r}; the artifact gives {real:.2f}x "
                    f"(graphrag-local {mean('graphrag-local'):.1f} ms / bm25 "
                    f"{mean('bm25'):.1f} ms)"
                )
        assert seen, (
            "no speed-ratio claim matched in README.md or results/RESULTS.md — this "
            "guard is bound to nothing, which is round 3's defect class (F6a). Either "
            "the ratio sentences were removed, or they are phrased in a shape this "
            "regex does not read: " + pattern.pattern
        )

    def test_per_case_table_is_bound(self) -> None:
        """The per-case table — the one that says the two manuscripts disagree.

        It is the evidence for "at this n no rung ordering survives per-case
        inspection", so it is the last table in the document that may drift. Each cell
        is located by row label (the rungs it names) and column header (the manuscript
        it names) and recomputed from `ablation-demo.json`'s own per-case rows, keyed
        by `case_doi`.
        """
        cases = _demo_cases()
        header, rows = _per_case_table()
        columns = {}
        for i, cell in enumerate(header[1:], start=1):
            doi = _case_of(cell, cases)
            if doi is not None:
                columns[i] = doi
        assert len(columns) >= 2, (
            f"the per-case table's headers {header[1:]} name "
            f"{len(columns)} of the {len(cases)} cases in ablation-demo.json — a column "
            "header must name the manuscript it reports, e.g. "
            f"{sorted({stem for stem in cases.values()})}"
        )
        data = {(str(r["rung"]), str(r["case_doi"])): r for r in _demo_rows()}
        compared = 0
        for row in rows:
            rungs = _rungs_named(row[0])
            assert rungs, (
                f"the per-case row labelled {row[0]!r} names no rung this code runs "
                f"({sorted(RUNG_ALIASES)})"
            )
            for i, doi in columns.items():
                assert i < len(row), f"row {row[0]!r} has no cell under {header[i]!r}"
                shown = _numbers(row[i])
                assert len(shown) == 2, (
                    f"{row[0]!r} x {header[i]!r}: expected 'recall / NDCG', got {row[i]!r}"
                )
                for rung in rungs:
                    record = data[(rung, doi)]
                    for value, field in zip(shown, ("recall_at_k", "ndcg_at_k"), strict=True):
                        assert value == pytest.approx(float(record[field]), abs=0.001), (  # type: ignore[arg-type]
                            f"per case, {rung} on {cases[doi]}: the table says "
                            f"{row[i]!r}, the artifact's {field} is {record[field]}"
                        )
                        compared += 1
        assert compared >= 8, (
            f"the per-case table bound only {compared} cells — too few for the table "
            "the aggregate is not allowed to be published without"
        )


def _demo_rows() -> list[dict[str, object]]:
    report = json.loads((ROOT / "results" / "ablation-demo.json").read_text())
    rows: list[dict[str, object]] = report["results"]
    return rows


def _demo_cases() -> dict[str, str]:
    """{preprint DOI: manuscript file stem} for every case the demo ladder ran."""
    ran = {str(r["case_doi"]) for r in _demo_rows()}
    cases: dict[str, str] = {}
    for path in sorted((ROOT / "manuscripts").glob("*.txt")):
        header, _body = read_manuscript(path)
        if header.preprint_doi in ran:
            cases[header.preprint_doi] = path.stem
    assert set(cases) == ran, (
        f"ablation-demo.json ran cases {sorted(ran)}; manuscripts/ accounts for "
        f"{sorted(cases)}"
    )
    return cases


def _case_of(header_cell: str, cases: dict[str, str]) -> str | None:
    """Which case a column header names, by a word of the manuscript's own filename."""
    text = _norm(header_cell)
    matched = {
        doi
        for doi, stem in cases.items()
        if any(word and word in text for word in stem.lower().split("-"))
    }
    assert len(matched) <= 1, f"the column header {header_cell!r} names {matched}"
    return matched.pop() if matched else None


def _rungs_named(label: str) -> list[str]:
    """Every rung a row label names — a row may cover two ("BM25 · RRF")."""
    parts = [p for p in re.split(r"[·,/]| and ", _norm(label)) if p.strip()]
    found: list[str] = []
    for part in parts:
        hit = [r for r, words in RUNG_ALIASES.items() if all(w in part for w in words)]
        assert len(hit) <= 1, f"{part!r} in {label!r} names more than one rung: {hit}"
        found.extend(hit)
    return found


def _per_case_table() -> tuple[list[str], list[list[str]]]:
    """The per-case table: one whose every data column reports recall AND NDCG together.

    That is what separates it from the aggregate table above it, which gives recall and
    NDCG their own columns — so the two can never be confused by position.
    """
    found = [
        (header, rows)
        for header, rows in _tables(RESULTS)
        if len(header) >= 2
        and all("recall" in _norm(c) and "ndcg" in _norm(c) for c in header[1:])
    ]
    assert len(found) == 1, (
        f"expected exactly one per-case table in results/RESULTS.md (every data column "
        f"headed 'recall / NDCG'); found {len(found)}. The aggregate mean is never "
        "published without the per-case rows it was computed from."
    )
    return found[0]



_NUMBER_WORDS = {
    "zero": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
}


def _quantity(token: str) -> float | None:
    """A spelled or written number, or None if this word is not one."""
    word = token.strip().lower()
    if word in _NUMBER_WORDS:
        return _NUMBER_WORDS[word]
    return float(word) if re.fullmatch(r"\d+(?:\.\d+)?", word) else None


def _gloss_claim(context: str) -> tuple[float, float] | None:
    """(low, high) claimed by the words immediately before a per-unit phrase.

    Both wordings are read: "five to six verdicts in ten" (a range, when the two runs
    disagree) and "three in ten" (one number, when they agree — which the committed
    runs now do). A noun may sit between the numbers and the phrase.
    """
    ranged = re.search(r"([\w.]+) (?:and|to) ([\w.]+)(?: \w+){0,2}[\s,]*$", context)
    if ranged:
        low, high = _quantity(ranged.group(1)), _quantity(ranged.group(2))
        if low is not None and high is not None:
            return low, high
    single = re.search(r"([\w.]+)(?: \w+){0,2}[\s,]*$", context)
    if single:
        value = _quantity(single.group(1))
        if value is not None:
            return value, value
    return None


class TestPanelGlossesMatchTheTable:
    """The plain-English sentences about swap-consistency, against the runs."""

    def _flipped(self) -> list[tuple[int, int]]:
        out = []
        for path in PANEL_RECORDS:
            data = json.loads(path.read_text())
            flipped = sum(1 for v in data["verdicts"] if not v["swap_consistent"])
            out.append((flipped, int(data["n_swap_checked"])))
        return out

    def test_per_twelve_and_per_ten_glosses(self) -> None:
        """Round-3 F6b: the old regex required the literal "in every ten" while the
        live gloss read "in ten", so a sentence claiming twice the measured flip rate
        sat in the closing paragraph, unguarded, listing the panel's strengths.

        The gloss is read from the words BEFORE the per-unit phrase rather than by a
        fixed sentence shape, because the two honest wordings differ: a range when the
        runs disagree ("five to six verdicts in ten") and a single number when they do
        not ("three in ten"). A regex that only reads a range goes dead the moment the
        runs agree, which is the same defect one wording later.
        """
        rates = [f / n for f, n in self._flipped()]
        lo, hi = min(rates), max(rates)
        seen = 0
        units = ((12, r"(?:of|in) (?:every )?twelve"), (10, r"(?:of|in) (?:every )?ten"))
        for unit, phrase in units:
            for match in re.finditer(phrase, RESULTS + README, re.IGNORECASE):
                context = (RESULTS + README)[max(0, match.start() - 60) : match.start()]
                claim = _gloss_claim(context)
                if claim is None:
                    continue
                seen += 1
                claimed_lo, claimed_hi = claim
                assert claimed_lo == pytest.approx(lo * unit, abs=0.5), (
                    f"'{context.strip()[-40:]} {match.group(0)}' understates: measured "
                    f"{lo * unit:.1f} per {unit}"
                )
                assert claimed_hi == pytest.approx(hi * unit, abs=0.5), (
                    f"'{context.strip()[-40:]} {match.group(0)}' overstates: measured "
                    f"{hi * unit:.1f} per {unit}"
                )
        assert seen, (
            "no per-ten or per-twelve gloss matched in README.md or results/RESULTS.md. "
            "The measured flip rate is "
            f"{lo:.2f}-{hi:.2f} ({lo * 10:.1f} to {hi * 10:.1f} in ten); a gloss of it "
            "must read like 'three in ten' or 'three to four verdicts in ten', or this "
            "guard covers nothing (round 3's F6b)."
        )

    def test_swap_rates_quoted_match_the_records(self) -> None:
        published = {float(m) for m in re.findall(r"swap-consistency (\d\.\d+)", RESULTS + README)}
        actual = set()
        for path in PANEL_RECORDS:
            data = json.loads(path.read_text())
            rate = data["swap_consistency_rate"]
            if rate is not None:
                actual.add(round(float(rate), 2))
        assert published, (
            "no swap-consistency rate is quoted in either file — the guard is bound to "
            f"nothing; the records carry {sorted(actual)}"
        )
        for value in published:
            assert round(value, 2) in actual, f"{value} is not any run's rate: {actual}"


def _build_stats(corpus: str) -> dict[str, object]:
    stats: dict[str, object] = json.loads(
        (ROOT / "results" / f"build-stats-{corpus}.json").read_text()
    )
    return stats


def _documents(corpus: str) -> int:
    return len(CorpusManifest.load(ROOT / "corpus" / f"{corpus}.manifest.json").docs)


def _minutes(cell: str) -> float:
    """A published wall-clock cell ("2 h 31 min", "45 min") in minutes."""
    text = _norm(cell)
    hours = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", text)
    mins = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b", text)
    assert hours or mins, f"{cell!r} carries no wall-clock figure"
    return (float(hours.group(1)) * 60 if hours else 0.0) + (
        float(mins.group(1)) if mins else 0.0
    )


class TestIndexTableMatchesBuildStats:
    """EVERY row of the index table, by row label, against the two build records.

    The table used to be bound on two rows (entities, edges) out of nine — the other
    seven were free to drift, including the wall-clock the README quotes and the
    per-call margin that is the whole run-conditions claim. Now each row is located by
    its label, each column by its corpus, and any row the table adds that this test
    does not know about fails: an unbound row in a bound table is the drift shape
    wearing a green tick.
    """

    def _table(self) -> tuple[dict[str, int], list[list[str]]]:
        """({corpus: column index}, rows) for the index table."""
        for header, rows in _tables(RESULTS):
            columns = {
                corpus: i
                for corpus in ("ci", "demo")
                for i, cell in enumerate(header)
                if _norm(cell) == f"{corpus} corpus"
            }
            if len(columns) == 2:
                return columns, rows
        raise AssertionError(
            "results/RESULTS.md has no index table: expected one whose header names "
            "'CI corpus' and 'demo corpus' as its two data columns"
        )

    def _expected(self, corpus: str) -> dict[str, tuple[float, ...]]:
        """{row-label keyword: the numbers that row must show} for one corpus."""
        stats = _build_stats(corpus)
        calls: dict[str, object] = stats["model_calls"]  # type: ignore[assignment]
        per_resolution: dict[str, int] = stats["communities_per_resolution"]  # type: ignore[assignment]
        return {
            "documents": (_documents(corpus),),
            "chunks": (float(stats["chunks"]),),  # type: ignore[arg-type]
            "entities": (float(stats["nodes"]),),  # type: ignore[arg-type]
            "edges": (float(stats["edges"]),),  # type: ignore[arg-type]
            "communities": (float(per_resolution["1.0"]),),
            "truncated": (float(stats["truncated_chunks"]),),  # type: ignore[arg-type]
            "calls": (float(calls["calls"]),),  # type: ignore[arg-type]
            "largest prompt": (
                float(calls["largest_prompt_tokens"]),  # type: ignore[arg-type]
                float(calls["smallest_headroom_tokens"]),  # type: ignore[arg-type]
            ),
            "wall-clock": (float(stats["extraction_wall_s"]) / 60,),  # type: ignore[arg-type]
        }

    def test_every_row_of_the_index_table_is_bound(self) -> None:
        columns, rows = self._table()
        keywords = list(self._expected("ci"))
        bound: set[str] = set()
        for row in rows:
            label = _norm(row[0])
            matched = [k for k in keywords if k in label]
            assert len(matched) == 1, (
                f"the index row {row[0]!r} matches {matched or 'no'} bound row(s); every "
                f"row of this table is bound to a build record, and the labels this test "
                f"reads are {keywords}"
            )
            keyword = matched[0]
            bound.add(keyword)
            for corpus, column in columns.items():
                assert column < len(row), f"row {row[0]!r} has no {corpus} cell"
                expected = self._expected(corpus)[keyword]
                cell = row[column]
                if keyword == "wall-clock":
                    shown: tuple[float, ...] = (_minutes(cell),)
                    assert abs(shown[0] - expected[0]) <= 1.0, (
                        f"{corpus} {keyword}: the table says {cell!r}, the record's "
                        f"extraction_wall_s is {expected[0] * 60:.0f} s "
                        f"({expected[0]:.0f} min)"
                    )
                    continue
                shown = tuple(_numbers(cell))
                assert shown == pytest.approx(expected), (
                    f"{corpus} {keyword}: the table says {cell!r}, "
                    f"build-stats-{corpus}.json says {expected}"
                )
        missing = [k for k in keywords if k not in bound]
        assert not missing, (
            f"the index table has no row for {missing}; a build record's field with no "
            "published row is a measurement nobody reads"
        )


# ======================================================================================
# The panel run's own table — round 5, report 3, Finding 1.
#
# `results/RESULTS.md`'s panel table was published unbound. An evaluator changed
# `0 SUPPORTS · 0 REFUTES · **20 NOT_ENOUGH_INFO**` to `2 SUPPORTS · 0 REFUTES ·
# **18 NOT_ENOUGH_INFO**` — the single most load-bearing honest number on the page, the
# one the section "the verifier decides nothing" is built on — and all 594 tests passed.
# Nine other published figures moved the same way.
#
# So this table is bound the way the index table above is: each row located by its LABEL,
# each value recomputed from `results/panel-review-*.json`, and a row the table gains that
# this test does not know about is a failure rather than a silence. Every figure is the
# figure ONE run produced: the two committed runs are byte-identical apart from wall-clock
# (report 1, Finding 4), so a table cell that summed them would state its n at twice its
# true value. Where the runs disagree, this fails and says so — one column cannot describe
# two different runs.
#
# Every checker takes the document as an argument so `TestThePanelTableBindingsCanGoRed`
# can run it over a mutated copy in memory and prove it fires. The repository's own rule,
# from round 4: a verification that cannot fail is worth nothing.
# ======================================================================================


def _panel_runs() -> list[tuple[str, dict[str, Any]]]:
    """(run label, record) for every committed panel run, ordered by run number.

    The first run's record carries no `-runN` suffix, so it is run 1 by construction —
    the same convention `make review` and `make review RUN=2` write.
    """
    runs: list[tuple[str, dict[str, Any]]] = []
    for path in PANEL_RECORDS:
        match = re.search(r"-run(\d+)$", path.stem)
        runs.append((match.group(1) if match else "1", json.loads(path.read_text())))
    labels = [label for label, _ in runs]
    assert len(set(labels)) == len(labels), (
        f"two committed panel records claim to be the same run: {labels}"
    )
    return sorted(runs, key=lambda run: int(run[0]))


def _agreed(figure: str, values: list[Any]) -> Any:
    """The one value every committed run produced, or a failure naming the disagreement.

    A single published cell describes every run. If the runs ever stop agreeing, the table
    has to grow a column before any number in it can be believed — so this fails rather
    than silently reporting the first run's figure.
    """
    distinct: list[Any] = []
    for value in values:  # by equality, not by hash: a scores mapping is not hashable
        if value not in distinct:
            distinct.append(value)
    assert len(distinct) == 1, (
        f"the committed panel runs disagree on {figure} ({values}); results/RESULTS.md "
        "publishes one value for all of them, so either the table needs a column per run "
        "or one of the records does not belong to this configuration"
    )
    return distinct[0]


def _verdict_counts(record: dict[str, Any]) -> dict[str, int]:
    return {v: sum(1 for row in record["verdicts"] if row["verdict"] == v) for v in VERDICTS}


def _reviewer_scores(record: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    return {
        str(output["reviewer"]): tuple(int(output["scores"][d]) for d in RUBRIC_DIMENSIONS)
        for output in record["reviewer_outputs"]
    }


# {binding: a pattern matching the row label that carries it}. Anchored where a looser
# word would match two rows: three labels contain "findings", and only one of them is the
# per-reviewer count.
_PANEL_ROWS = {
    "reviewers": r"^reviewers$",
    "findings per reviewer": r"^findings$",
    "unparsed findings": r"unparseable|failed to parse",
    "citations dropped": r"citations? (?:deleted|dropped|removed)",
    "malformed findings": r"malformed",
    "rubric scores": r"^scores$",
    "claims verified": r"claims verified",
    "verdict distribution": r"^verdicts$",
    "grounded verdicts": r"evidence span",
    "conflicts": r"conflicts",
    "deterministic lens": r"lens",
}


def _panel_table(text: str) -> list[list[str]]:
    """The panel table: the label/value table whose rows include the verdict distribution.

    Located by shape and by content, never by position — the document has ten other pipe
    tables and gains one most rounds.
    """
    found = [
        rows
        for header, rows in _tables(text)
        if len(header) == 2
        and not any(_norm(c) for c in header)
        and any(re.fullmatch(r"verdicts", _norm(row[0])) for row in rows if row)
    ]
    assert len(found) == 1, (
        f"expected exactly one panel table in results/RESULTS.md (two unheaded columns, a "
        f"row labelled 'verdicts'); found {len(found)}. That table is where the verdict "
        "distribution, the grounding yield and both hygiene counters are published."
    )
    return found[0]


def _check_the_panel_table(text: str) -> None:
    """Every row of the panel table, by label, against every committed panel record."""
    runs = _panel_runs()
    rows = _panel_table(text)
    bound: dict[str, str] = {}
    for row in rows:
        label = _norm(row[0])
        matched = [name for name, pattern in _PANEL_ROWS.items() if re.search(pattern, label)]
        assert len(matched) == 1, (
            f"the panel row {row[0]!r} matches {matched or 'no'} binding(s); every row of "
            f"this table is recomputed from the panel records, and the bindings this test "
            f"carries are {sorted(_PANEL_ROWS)}. An unbound row in a bound table is the "
            "drift shape wearing a green tick."
        )
        assert len(row) >= 2, f"the panel row {row[0]!r} has no value cell"
        assert matched[0] not in bound, f"two panel rows bind {matched[0]!r}"
        bound[matched[0]] = row[1]
    missing = sorted(set(_PANEL_ROWS) - set(bound))
    assert not missing, (
        f"the panel table has no row for {missing}. Each of those is a figure the records "
        "carry and the prose leans on; a row that leaves the table stops being checked."
    )

    def numbers(binding: str) -> list[float]:
        return _numbers(bound[binding])

    def expect(binding: str, expected: tuple[float, ...], how: str) -> None:
        shown = tuple(numbers(binding))
        assert shown == pytest.approx(expected), (
            f"panel table, {binding}: the table says {bound[binding]!r}, the committed "
            f"records say {expected} ({how})"
        )

    expect(
        "reviewers",
        (float(_agreed("reviewer count", [len(r["reviewer_outputs"]) for _, r in runs])),),
        "reviewer_outputs",
    )
    expect(
        "findings per reviewer",
        (
            float(
                _agreed(
                    "findings per reviewer",
                    [len(o["findings"]) for _, r in runs for o in r["reviewer_outputs"]],
                )
            ),
        ),
        "len(findings) on every reviewer of every run — the cell says 'each', so a run "
        "whose reviewers wrote different numbers cannot be published this way",
    )
    expect(
        "unparsed findings",
        (
            float(
                _agreed(
                    "unparsed reviewers",
                    [sum(1 for o in r["reviewer_outputs"] if o["truncated"]) for _, r in runs],
                )
            ),
        ),
        "reviewer_outputs[].truncated",
    )
    for binding, field in (
        ("citations dropped", "unretrieved_citations_dropped"),
        ("malformed findings", "malformed_findings_dropped"),
    ):
        expect(
            binding,
            (float(_agreed(field, [o[field] for _, r in runs for o in r["reviewer_outputs"]])),),
            f"{field} on every reviewer of every run",
        )
    expect(
        "claims verified",
        (float(_agreed("verdict count", [len(r["verdicts"]) for _, r in runs])),),
        "len(verdicts) in one run",
    )
    expect(
        "grounded verdicts",
        (
            float(
                _agreed(
                    "grounded verdicts",
                    [sum(1 for v in r["verdicts"] if v["evidence"]) for _, r in runs],
                )
            ),
            float(_agreed("verdict count", [len(r["verdicts"]) for _, r in runs])),
        ),
        "verdicts carrying an evidence span, of the verdicts in ONE run — the pair the "
        "records support; 12 of 40 counts the same 20 verdicts twice",
    )
    expect(
        "conflicts",
        (float(_agreed("conflicts", [len(r["conflicts"]) for _, r in runs])),),
        "len(conflicts)",
    )
    expect(
        "deterministic lens",
        (float(_agreed("lens findings", [len(r["deterministic_findings"]) for _, r in runs])),),
        "len(deterministic_findings)",
    )

    # The claims-verified cell also promises every claim was judged twice. That is a
    # property of the rows, not a number, so it is checked against them rather than left
    # as a sentence nobody reads.
    if "twice" in _norm(bound["claims verified"]):
        for label, record in runs:
            unswapped = [v["claim_id"] for v in record["verdicts"] if not v["swapped"]]
            assert not unswapped, (
                f"the panel table says every claim was judged twice; run {label} records "
                f"{len(unswapped)} verdict(s) that were not swapped ({unswapped[:3]})"
            )
            assert record["n_swap_checked"] == len(record["verdicts"]), (
                f"run {label}: n_swap_checked is {record['n_swap_checked']} over "
                f"{len(record['verdicts'])} verdicts, so 'every one judged twice' is not "
                "what the record says"
            )

    _check_the_verdict_distribution_cell(bound["verdict distribution"], runs)
    _check_the_rubric_scores_cell(bound["rubric scores"], runs)


def _check_the_verdict_distribution_cell(cell: str, runs: list[tuple[str, dict[str, Any]]]) -> None:
    """`0 SUPPORTS · 0 REFUTES · **20 NOT_ENOUGH_INFO**` — the cell the evaluator moved.

    Every verdict the schema allows must appear with its count, so a class cannot be
    dropped from the table the moment it goes to zero — or, as the mutation did, moved
    into another class. The names are the code's own constants, matched case-sensitively:
    `VERDICTS` is what the records are validated against.
    """
    published = {
        verdict: int(shown.replace(",", ""))
        for shown, verdict in re.findall(r"(\d[\d,]*)\s*\**\s*(" + "|".join(VERDICTS) + r")", cell)
    }
    assert set(published) == set(VERDICTS), (
        f"the verdict row reads {cell!r}; it must publish a count for every verdict this "
        f"code can emit ({list(VERDICTS)}), because a class that leaves the table when it "
        f"reaches zero takes its number with it. Read: {sorted(published)}"
    )
    for verdict in VERDICTS:
        expected = _agreed(f"{verdict} count", [_verdict_counts(r)[verdict] for _, r in runs])
        assert published[verdict] == expected, (
            f"the verdict row says {published[verdict]} {verdict}; every committed panel "
            f"record says {expected} in one run. The row reads {cell!r}"
        )


def _check_the_rubric_scores_cell(cell: str, runs: list[tuple[str, dict[str, Any]]]) -> None:
    """`methods 4/4/4 · novelty 4/4/**3**` — each reviewer's scores, in rubric order."""
    scores = _agreed("reviewer scores", [_reviewer_scores(record) for _, record in runs])
    parts = [part for part in re.split(r"[·|;]", cell) if _numbers(part)]
    seen: set[str] = set()
    for part in parts:
        word = _norm(part).split()[0]
        matched = [name for name in scores if word in name.lower()]
        assert len(matched) == 1, (
            f"the scores cell segment {part.strip()!r} opens with {word!r}, which names "
            f"{matched or 'no'} reviewer of {sorted(scores)}"
        )
        shown = tuple(_numbers(part))
        assert shown == pytest.approx(tuple(float(v) for v in scores[matched[0]])), (
            f"the scores cell says {part.strip()!r}; {matched[0]} recorded "
            f"{scores[matched[0]]} for {RUBRIC_DIMENSIONS}"
        )
        seen.add(matched[0])
    assert seen == set(scores), (
        f"the scores cell publishes {sorted(seen)}; the records carry {sorted(scores)}, "
        "and a reviewer whose scores leave the table stops being checked"
    )


def _swap_table(text: str) -> tuple[list[str], list[list[str]]]:
    """The per-run swap table: header names a run column and a swap-consistency column."""
    found = [
        ([_norm(c) for c in header], rows)
        for header, rows in _tables(text)
        if any(_norm(c) == "run" for c in header)
        and any("swap" in _norm(c) for c in header)
    ]
    assert len(found) == 1, (
        f"expected exactly one per-run swap table in results/RESULTS.md (a 'run' column "
        f"and a swap-consistency column); found {len(found)}"
    )
    return found[0]


def _check_the_swap_table(text: str) -> None:
    """Every run's own row — the honest per-run form of the figures the prose doubles.

    Report 1's Finding 4: `12 of 40` and `0 of 32` count one population twice. This table
    is where the repository already does it right, one row per run, so it is bound row by
    row and every committed record must have a row: a run that is published in aggregate
    and nowhere per run is exactly the shape the finding is about.
    """
    header, rows = _swap_table(text)
    runs = dict(_panel_runs())
    columns = {
        key: index
        for key, words in (
            ("consistent", ("swap-consistent",)),
            ("abstain", ("abstain",)),
            ("rate", ("rate",)),
            ("tokens", ("tokens",)),
            ("wall", ("wall",)),
        )
        for index, cell in enumerate(header)
        if all(word in cell for word in words)
    }
    missing = sorted({"consistent", "abstain", "rate", "tokens", "wall"} - set(columns))
    assert not missing, f"the swap table has no column for {missing}; its headers are {header}"

    published: set[str] = set()
    for row in rows:
        label = _numbers(row[0])
        assert len(label) == 1, f"the swap row {row[0]!r} does not name one run"
        run = str(int(label[0]))
        assert run in runs, (
            f"the swap table publishes a row for run {run}; the committed records are "
            f"{sorted(runs)}. A published run with no record is a number from nowhere."
        )
        published.add(run)
        record = runs[run]
        verdicts = record["verdicts"]
        consistent = sum(1 for v in verdicts if v["swap_consistent"])
        checked = int(record["n_swap_checked"])
        for key, expected in (
            ("consistent", (float(consistent), float(checked))),
            ("abstain", (float(len(verdicts) - consistent), float(checked))),
            ("tokens", (float(record["total_tokens"]),)),
        ):
            shown = tuple(_numbers(row[columns[key]]))
            assert shown == pytest.approx(expected), (
                f"swap table, run {run}, {key}: the table says "
                f"{row[columns[key]]!r}, the record says {expected}"
            )
        shown_rate = re.search(r"\d+(?:\.\d+)?", _norm(row[columns["rate"]]))
        assert shown_rate, f"swap table, run {run}: no rate in {row[columns['rate']]!r}"
        assert float(shown_rate.group(0)) == pytest.approx(
            float(record["swap_consistency_rate"]), abs=_slack(shown_rate.group(0))
        ), (
            f"swap table, run {run}: the table says swap-consistency "
            f"{shown_rate.group(0)}, the record says {record['swap_consistency_rate']}"
        )
        wall = tuple(_numbers(row[columns["wall"]]))
        assert wall == pytest.approx((float(record["wall_s"]),), abs=0.05), (
            f"swap table, run {run}: the table says {row[columns['wall']]!r}, the record's "
            f"wall_s is {record['wall_s']}"
        )
    assert published == set(runs), (
        f"the swap table publishes runs {sorted(published)}; the committed records are "
        f"{sorted(runs)}. Every run this repository ships gets its own row, because the "
        "aggregate over two byte-identical runs is a reproduction, not a sample."
    )


_CHUNKS = re.compile(r"(\d[\d,]*)\s+chunks", re.IGNORECASE)
# What the page cannot stop saying and still be describing the exclusion. Keyed on the
# stem rather than on `drop`/`withheld` as written, so "26 chunks were removed" stays in
# scope — round 6's rule: a guard keyed to a sentence is evaded by rewriting it.
_EXCLUSION_VERB = re.compile(r"drop|withh|withheld|exclu|remov", re.IGNORECASE)


def _dropped_chunks_by_subject() -> dict[str, set[int]]:
    """{manuscript stem: the chunk counts exclusion dropped}, over every committed record.

    Both record families carry it — the panel run per run, the planted evaluation per arm
    — and the prose quotes the figure in both places ("26 chunks dropped, recorded in the
    record", "26 chunks dropped on every arm").
    """
    counts: dict[str, set[int]] = defaultdict(set)
    for path in PANEL_RECORDS:
        stem = re.sub(r"-run\d+$", "", path.stem.replace("panel-review-", "", 1))
        counts[stem].add(int(json.loads(path.read_text())["dropped_chunks"]))
    for path in sorted((ROOT / "results").glob("planted-eval-*.json")):
        record = json.loads(path.read_text())
        stem = Path(str(record["manuscript"])).stem
        for arm in record["results"]:
            counts[stem].add(int(arm["dropped_chunks"]))
    assert counts, "no committed record carries an exclusion figure"
    return dict(counts)


def _check_the_exclusion_figures(results_text: str, readme_text: str) -> None:
    """"26 chunks dropped" — the self-exclusion figure, wherever either file states it.

    Bound to the manuscript the sentence names where it names one; where it does not, the
    number still has to be one a committed record produced. That is weaker, and it is
    weaker in the one direction that cannot hide a wrong number: the evaluator's `26` ->
    `46` fails either way.
    """
    by_subject = _dropped_chunks_by_subject()
    everything = {count for counts in by_subject.values() for count in counts}
    for where, text in (("results/RESULTS.md", results_text), ("README.md", readme_text)):
        seen = 0
        for match in _CHUNKS.finditer(text):
            window = text[max(0, match.start() - 200) : match.end() + 120].lower()
            if not _EXCLUSION_VERB.search(window):
                continue
            seen += 1
            shown = int(match.group(1).replace(",", ""))
            named = [stem for stem in by_subject if stem.split("-")[0].lower() in window]
            expected = by_subject[named[0]] if len(named) == 1 else everything
            assert shown in expected, (
                f"{where} says {match.group(0)!r} were dropped"
                + (f" for {named[0]}" if len(named) == 1 else "")
                + f"; the committed records say {sorted(expected)}.\n  "
                + re.sub(r"\s+", " ", text[max(0, match.start() - 120) : match.end() + 80])
            )
        assert seen, (
            f"{where} states the exclusion figure in no form this rule can read, so the "
            f"guard covers nothing here (round 3's F6). The records carry {by_subject}, "
            "and self-exclusion is one of the four mechanisms both documents present as "
            "working. `seen` is counted per document deliberately: counted across both, "
            "one file's sentence keeps the other file's guard looking alive, which is "
            "the half of round 6's finding 2 that had nothing to do with the regex."
        )


# ======================================================================================
# Two pages whose promises outran their bindings (round 6, report 1, findings 3 and 4).
#
# Round 6's rule, and the reason both rules below are written the way they are: a guard
# keyed to a SENTENCE is evaded by rewriting the sentence, so each of these keys on the
# NUMBER and the ROLE it plays — a count of entities the rebuild removed, a population of
# committed reports — and reads it wherever the page puts it. A page that states one of
# these figures in a form the rule cannot read fails here as unreadable; it never passes
# in silence, which is the whole point of the repair.
# ======================================================================================

# A sentence break: a full stop followed by something that starts a sentence. Naive
# splitting on `[.!?]\s+` cuts `tests/test_run_graph.py verifies…` in half, and half a
# sentence is half a scope.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")
_HISTORICAL_CLAIM = re.compile(
    r"an earlier (?:version|run|record|reading)|a previous version|previous version of this|"
    r"the figure published until this commit|used to (?:say|read|report)|was once",
    re.IGNORECASE,
)


def _live_sentences(text: str) -> list[tuple[str, str]]:
    """(the paragraph a sentence sits in, the sentence) — for the live claims only.

    Two scopes, for the reason `test_artifact_conformance.py` gives: a VALUE is a
    property of the sentence making the claim, so a paragraph must never lend a figure
    three sentences away its caveat; but where a number came FROM is often established
    one sentence later, so a provenance is looked for in the paragraph.

    Formatting goes (a figure wrapped in `**` would otherwise never end a sentence), and
    a sentence that says in so many words that it is reporting a superseded reading is
    not bound to today's records — publishing the old number beside the correction is
    something this repository does on purpose.
    """
    flat = re.sub(r"[*`]", "", text)
    out: list[tuple[str, str]] = []
    for block in re.split(r"\n\s*\n", flat):
        joined = re.sub(r"\s+", " ", block).strip()
        if not joined:
            continue
        out += [
            (joined, sentence)
            for sentence in _SENTENCE_BREAK.split(joined)
            if sentence.strip() and not _HISTORICAL_CLAIM.search(sentence)
        ]
    return out


def _the_excluded_twin() -> str:
    """The published twin the committed runs withhold, read from the records.

    Never hardcoded: the id is what `excluded_docs` says, so a corpus that changed its
    twin moves this binding with it rather than measuring the wrong document silently.
    """
    named = {
        doc
        for path in PANEL_RECORDS
        for doc in json.loads(path.read_text())["excluded_docs"]
    }
    assert len(named) == 1, (
        f"the committed panel runs withhold {sorted(named)}; LIMITATIONS.md publishes one "
        "rebuild residue, and that is only one measurement while there is one twin"
    )
    return str(named.pop())


@cache
def _ci_rebuild_residue() -> tuple[int, int, float]:
    """(entities removed, edges lightened, weight stripped) on the CI corpus.

    Recomputed offline from `fixtures/extraction/ci` by the same route
    `tests/test_run_graph.py` drives — a pure merge over cached extractions, no model and
    no network. That file proves the rebuild is CORRECT by deriving its expectation from
    an independent mechanism and deliberately pins no constant; this one binds the
    published PROSE to what the rebuild actually does, which is the half `LIMITATIONS.md:7`
    promises and round 6 found missing. Cached: two graph builds serve every assertion.
    """
    twin = _the_excluded_twin()
    full, _fa, withheld_none = build_run_graph(ROOT, "ci", set())
    run, _ra, withheld = build_run_graph(ROOT, "ci", {twin})
    assert withheld_none == 0 and withheld > 0, (
        f"the CI rebuild withheld {withheld} chunks for {twin}: the residue this page "
        "measures is the residue of an exclusion that happened, so a rebuild that "
        "withholds nothing is a finding rather than a figure to publish"
    )
    lighter = [
        (a, b) for a, b in run.edges() if run.edges[a, b]["weight"] < full.edges[a, b]["weight"]
    ]
    stripped = sum(
        full.edges[a, b]["weight"] - run.edges[a, b]["weight"] for a, b in lighter
    )
    return len(set(full) - set(run)), len(lighter), float(stripped)


# Each role, as the page cannot state the figure without naming it. The value is read
# wherever the role puts it — "removes 142 entities", "142 entities are removed",
# "strips 27.25 units … from 48 edges" — never by the sentence around it.
_RESIDUE_ROLES = (
    ("entities removed", re.compile(r"(?P<n>\d[\d,]*)\s+entit(?:y|ies)\b", re.IGNORECASE)),
    ("edges lightened", re.compile(r"(?P<n>\d[\d,]*)\s+edges?\b", re.IGNORECASE)),
    (
        "weight stripped",
        re.compile(r"(?P<n>\d[\d,]*(?:\.\d+)?)\s+units?\b", re.IGNORECASE),
    ),
)
# The claim is about the rebuild on the CI corpus. Both halves have to be in the
# sentence: the page measures a second residue on the MET17 run ("101 graph entities
# exist only because of the excluded twin"), and a scope that swept every "N entities"
# would bind that one to this measurement.
_CI_SCALE = re.compile(r"\bCI corpus\b|corpus/ci\b", re.IGNORECASE)
_REBUILD_VERB = re.compile(r"remov\w*|strip\w*|withh\w*|exclud\w*", re.IGNORECASE)


def _locate_residue_claim(text: str, role: str) -> tuple[int, int, str] | None:
    """(start, end, value) of the first in-scope statement of `role` on the real page.

    The controls below mutate what the rule ACTUALLY reads, found the way the rule finds
    it, so a control can never move a string the page has stopped carrying and call the
    green that follows a proof.
    """
    pattern = dict(_RESIDUE_ROLES)[role]
    flat_sentences = [sentence for _paragraph, sentence in _live_sentences(text)]
    for sentence in flat_sentences:
        if not (_CI_SCALE.search(sentence) and _REBUILD_VERB.search(sentence)):
            continue
        match = pattern.search(sentence)
        if not match:
            continue
        # Back to the real bytes: the sentence is whitespace-collapsed, the file is not.
        needle = re.compile(
            r"\b" + re.escape(match.group("n")).replace(r"\ ", r"\s+") + r"\s+"
            + match.group(0).split(maxsplit=1)[1].replace(" ", r"\s+")
        )
        located = needle.search(text)
        if located:
            value = re.match(r"\S+", located.group(0))
            assert value is not None
            return located.start(), located.start() + len(value.group(0)), value.group(0)
    return None


def _check_the_ci_exclusion_residue(limitations_text: str) -> int:
    """LIMITATIONS.md's 142 / 48 / 27.25, against the graph rebuilt from committed bytes.

    Round 6, report 1, finding 3. `LIMITATIONS.md:7` promises that "a number that drifts
    from the measurement fails the suite rather than sitting here unread"; these three
    were carried in `tests/test_run_graph.py`'s DOCSTRINGS only, so 142 -> 143 and
    27.25 -> 27.75 both left the suite green. All three values were correct — what was
    wrong was the page's account of its own discipline, in the bullet describing the
    residue exclusion cannot reach, on the page the README calls the one most worth
    reading carefully.
    """
    removed, lightened, stripped = _ci_rebuild_residue()
    expected = {
        "entities removed": float(removed),
        "edges lightened": float(lightened),
        "weight stripped": stripped,
    }
    seen: dict[str, int] = {role: 0 for role, _pattern in _RESIDUE_ROLES}
    for _paragraph, sentence in _live_sentences(limitations_text):
        if not (_CI_SCALE.search(sentence) and _REBUILD_VERB.search(sentence)):
            continue
        for role, pattern in _RESIDUE_ROLES:
            for match in pattern.finditer(sentence):
                seen[role] += 1
                shown = match.group("n")
                assert float(shown.replace(",", "")) == pytest.approx(
                    expected[role], abs=_slack(shown)
                ), (
                    f"LIMITATIONS.md says {match.group(0).strip()!r}; rebuilding the CI "
                    f"graph from fixtures/extraction/ci with {_the_excluded_twin()} "
                    f"withheld gives {role} = {expected[role]:g}.\n  {sentence}"
                )
    missing = sorted(role for role, count in seen.items() if not count)
    assert not missing, (
        f"LIMITATIONS.md states {missing} in no form this rule can read, so the figure(s) "
        "it publishes for the CI rebuild are bound by nothing — which is the finding this "
        "rule closes, not a reason to pass. The rebuild removes "
        f"{removed} entities and strips {stripped:g} units of twin-contributed weight "
        f"from {lightened} edges; write each number beside its noun, in a sentence that "
        "names the CI corpus and what the rebuild does to it."
    )
    return sum(seen.values())


# The output cap where RESULTS.md restates it inside a TABLE. `test_artifact_conformance`
# binds the cap in prose and cannot reach here: it reads prose units, and a table row is
# deliberately not one of them (folding eleven unrelated cells into a "sentence" would let
# a claim borrow the caveat of a row three lines away). So the same number is bound here,
# per cell, by the constant it names — the asymmetry table's "MAX_FINDINGS 8 x 2
# reviewers = 16" restates three record values and was read by nothing.
_MAX_FINDINGS_CELL = re.compile(r"MAX_FINDINGS")


def _cap_cells(text: str) -> list[str]:
    """Every table cell that names the cap, as the rule below finds them."""
    return [
        cell
        for _header, rows in _tables(text)
        for row in rows
        for cell in row
        if _MAX_FINDINGS_CELL.search(cell)
    ]


def _check_the_cap_where_a_table_restates_it(results_text: str) -> int:
    """Every cell naming `MAX_FINDINGS` carries the cap, the reviewer count, or their
    product — and nothing else."""
    runs = _panel_runs()
    cap = _agreed(
        "findings per reviewer",
        [len(o["findings"]) for _label, r in runs for o in r["reviewer_outputs"]],
    )
    reviewers = _agreed("reviewer count", [len(r["reviewer_outputs"]) for _label, r in runs])
    allowed = {float(cap), float(reviewers), float(cap * reviewers)}
    seen = 0
    for _header, rows in _tables(results_text):
        for row in rows:
            for cell in row:
                if not _MAX_FINDINGS_CELL.search(cell):
                    continue
                seen += 1
                shown = _numbers(cell)
                assert shown, (
                    f"a results/RESULTS.md cell names MAX_FINDINGS and states no number "
                    f"this rule can read: {cell!r}. The cap is {cap} findings per "
                    f"reviewer over {reviewers} reviewers; write the figure beside the "
                    "constant, or the restatement is bound by nothing."
                )
                unexpected = [n for n in shown if n not in allowed]
                assert not unexpected, (
                    f"a results/RESULTS.md cell restating MAX_FINDINGS carries "
                    f"{unexpected}: {cell!r}. The committed runs give {cap} findings per "
                    f"reviewer, {reviewers} reviewers and {cap * reviewers} in a run."
                )
    assert seen, (
        "no results/RESULTS.md table cell names MAX_FINDINGS, so this rule covers nothing "
        "(round 3's F6). The panel table publishes the cap as its findings row and the "
        "asymmetry table restates it as the output ceiling; a cell that leaves the table "
        "stops being checked, and that is what has to be noticed."
    )
    return seen


# The demo community reports, as committed. `README.md` in that directory is the
# fixture's own note, not a report — the population is the records themselves.
def _committed_demo_reports() -> int:
    reports = sorted((ROOT / "fixtures" / "summaries" / "demo").glob("*.json"))
    assert reports, (
        "fixtures/summaries/demo holds no committed community report, so the population "
        "results/RESULTS.md states its model-drift figure over has nothing behind it"
    )
    return len(reports)


_DRIFT_SUBJECT = re.compile(r"communit\w*\s+reports?|summar(?:y|ies|iser)", re.IGNORECASE)
_DRIFT_ACT = re.compile(r"regenerat\w*|\bcold\b|reproducib\w*|reproduce[sd]?\b", re.IGNORECASE)
_PAIR = re.compile(r"(?P<num>\d[\d,]*)\s+of\s+(?P<den>\d[\d,]*)\b")
_REPORT_POPULATION = re.compile(
    r"(?P<n>\d[\d,]*)\s+(?:\w+\s+){0,2}?communit\w*\s+reports?\b", re.IGNORECASE
)
# What a reader can follow to the measurement itself. Either two commits they can
# `git show` — the two generations being compared — or a committed record naming the
# reports. Nothing else is a route, and the numerator is not recomputable offline: a
# second cold generation is a fresh draw from the model, not a replay.
_COMMITISH = r"\b(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b"
_ATTRIBUTION = (
    re.compile(rf"{_COMMITISH}[^.]{{0,300}}?{_COMMITISH}"),
    re.compile(r"results/[\w.-]+\.json"),
)


def _check_the_model_drift_figure(results_text: str) -> int:
    """"1 of 79" — the measured size of the model-layer reproducibility gap.

    Round 6, report 1, finding 4. The README sends a reader here for THE answer to what
    no replay can catch, and `results/RESULTS.md:3` promises every number on the page is
    read from the JSON record beside it. This one was read from no record: only one
    generation of the 79 reports is committed, no test mentioned the figure, and the only
    named route (`make demo-summaries`) needs Ollama and is the stochastic process being
    measured rather than a check of this reading.

    Half of it binds and half of it cannot, and this rule is exactly that honest. The
    POPULATION is a committed artifact — `fixtures/summaries/demo` — so it is recomputed
    here and a drift in it fails. The NUMERATOR is a comparison against a generation this
    repository does not commit, so no offline route can reproduce it; what this rule
    requires instead is the route a reader CAN follow — the two commits the comparison
    was made between, which `git show` will hand them, or a committed record that carries
    it. That is the shape `results/evaluation-loop.json` already uses for the one other
    number an outsider cannot recompute, and the page says so in the sentence that uses
    it. An unattributed number under a page-wide binding promise is the finding.
    """
    reports = _committed_demo_reports()
    checked = 0
    for paragraph, sentence in _live_sentences(results_text):
        if not (_DRIFT_SUBJECT.search(sentence) and _DRIFT_ACT.search(sentence)):
            continue
        pairs = list(_PAIR.finditer(sentence))
        populations = list(_REPORT_POPULATION.finditer(sentence))
        if not pairs and not populations:
            continue
        for match in populations:
            assert int(match.group("n").replace(",", "")) == reports, (
                f"results/RESULTS.md says {match.group(0).strip()!r}; "
                f"fixtures/summaries/demo commits {reports} community reports.\n  {sentence}"
            )
        for match in pairs:
            assert int(match.group("den").replace(",", "")) == reports, (
                f"results/RESULTS.md states the model-layer drift as "
                f"{match.group(0).strip()!r}; fixtures/summaries/demo commits {reports} "
                f"community reports, so that is the population.\n  {sentence}"
            )
            checked += 1
            assert any(pattern.search(paragraph) for pattern in _ATTRIBUTION), (
                "results/RESULTS.md publishes "
                f"{match.group(0).strip()!r} as the measured size of the model-layer "
                "reproducibility gap, and no committed byte can reproduce it: only one "
                f"generation of the {reports} demo community reports is committed, and "
                "regenerating them is a fresh draw from the model rather than a replay. "
                "The page opens by promising every number on it is read from the record "
                "beside it, so this number needs the route a reader can actually take, "
                "in the paragraph that states it — the two commits the two generations "
                "were compared between, so `git show <a>:fixtures/summaries/demo/<file>` "
                "and `git show <b>:…` settle it (at this commit `git diff --name-only "
                "52238c8 c607ff4 -- fixtures/summaries/demo` returns exactly the one "
                "file that moved), or a committed `results/*.json` record carrying the "
                "comparison. Attribution is the honest road here, the way "
                "results/evaluation-loop.json labels the one other number an outsider "
                f"cannot recompute.\n  {sentence}"
            )
    assert checked, (
        "results/RESULTS.md states the model-layer drift figure in no form this rule can "
        f"read. The committed population is {reports} demo community reports; write the "
        "figure as 'N of M' in a sentence that names the reports and the regeneration, "
        "so the population is bound and the claim is visible — a number stated in a form "
        "no rule can read is bound by nothing, which is round 6's finding 2."
    )
    return checked


class TestPanelTableMatchesTheRecords:
    def test_every_row_of_the_panel_table_is_bound(self) -> None:
        _check_the_panel_table(RESULTS)

    def test_every_run_has_its_own_swap_row(self) -> None:
        _check_the_swap_table(RESULTS)

    def test_the_exclusion_figures_are_bound(self) -> None:
        _check_the_exclusion_figures(RESULTS, README)


class TestTheLimitationsFiguresAreBound:
    """LIMITATIONS.md:7 says a figure that drifts fails the suite. Now three more do."""

    def test_the_ci_rebuild_residue_is_bound(self) -> None:
        assert _check_the_ci_exclusion_residue(LIMITATIONS) >= 3

    @pytest.mark.parametrize("role", [role for role, _pattern in _RESIDUE_ROLES])
    def test_a_moved_residue_figure_fails(self, role: str) -> None:
        """The evaluator's own mutations, from round 6, report 1, finding 3 — 142 -> 143
        and 27.25 -> 27.75 each left the whole suite green before this binding existed.

        The figure is LOCATED by the rule's own reader rather than pasted in as a string,
        so a control can never quietly mutate a sentence the page no longer carries: the
        first in-scope occurrence of this role is found on the real page and bumped.
        """
        found = _locate_residue_claim(LIMITATIONS, role)
        assert found, (
            f"LIMITATIONS.md states no {role} figure this rule can read, so this control "
            "mutates nothing; the binding above says so too, and that is the finding"
        )
        start, end, shown = found
        moved = f"{float(shown) + 1:g}" if "." not in shown else f"{float(shown) + 0.5:g}"
        with pytest.raises(AssertionError, match=r"rebuilding the CI graph"):
            _check_the_ci_exclusion_residue(LIMITATIONS[:start] + moved + LIMITATIONS[end:])

    def test_a_residue_figure_written_unreadably_fails(self) -> None:
        """Round 6's rule: a figure written in a form the rule cannot read must fail as
        unreadable, never pass because the parser shrugged. Here the number is deleted
        from the claim and its noun left standing."""
        found = _locate_residue_claim(LIMITATIONS, "entities removed")
        assert found, "LIMITATIONS.md states no entity-removal figure to erase"
        start, end, _shown = found
        with pytest.raises(AssertionError, match=r"in no form this rule can read"):
            _check_the_ci_exclusion_residue(LIMITATIONS[:start] + "some" + LIMITATIONS[end:])


class TestTheCapIsBoundWhereATableRestatesIt:
    """`MAX_FINDINGS` restated in a table cell — the reach `test_artifact_conformance`'s
    prose rule structurally does not have."""

    def test_every_cell_naming_the_cap_is_bound(self) -> None:
        assert _check_the_cap_where_a_table_restates_it(RESULTS) >= 2

    @pytest.mark.parametrize("index", [0, 1])
    def test_a_moved_cap_in_a_table_fails(self, index: int) -> None:
        """Each cell that restates the cap, mutated where the rule actually reads it.

        The cell is LOCATED on the real page rather than pasted in as a literal, so this
        control cannot quietly move a row the document no longer carries — and it does
        not have to spell the asymmetry table's multiplication sign back at it.
        """
        cells = _cap_cells(RESULTS)
        assert len(cells) > index, (
            f"results/RESULTS.md restates MAX_FINDINGS in {len(cells)} table cell(s); "
            "this control needs the cell it was written for, so re-point it or drop it"
        )
        cell = cells[index]
        digits = re.search(r"\d+", cell)
        assert digits, f"the cell {cell!r} carries no number to move"
        moved = cell[: digits.start()] + str(int(digits.group(0)) + 1) + cell[digits.end() :]
        with pytest.raises(AssertionError, match=r"restating MAX_FINDINGS carries"):
            _check_the_cap_where_a_table_restates_it(RESULTS.replace(cell, moved, 1))

    def test_a_cap_cell_with_no_number_fails(self) -> None:
        """A restatement written in words is a restatement nothing can read."""
        cell = _cap_cells(RESULTS)[0]
        wordy = re.sub(r"\d+", "a few", cell)
        assert wordy != cell, f"the cell {cell!r} carries no number to erase"
        with pytest.raises(AssertionError, match=r"states no number this rule can read"):
            _check_the_cap_where_a_table_restates_it(RESULTS.replace(cell, wordy, 1))


class TestTheModelDriftFigureIsBacked:
    """results/RESULTS.md's "1 of 79" — bound where it can be, attributed where it
    cannot (round 6, report 1, finding 4)."""

    def test_the_drift_figure_is_bound_and_attributed(self) -> None:
        assert _check_the_model_drift_figure(RESULTS) >= 1

    def test_a_moved_population_fails(self) -> None:
        original = "1 of 79**"
        assert original in RESULTS, f"results/RESULTS.md no longer contains {original!r}"
        with pytest.raises(AssertionError, match=r"community reports, so that is the"):
            _check_the_model_drift_figure(RESULTS.replace(original, "1 of 78**", 1))

    def test_a_moved_report_count_fails(self) -> None:
        original = "all 79 demo community reports"
        assert original in RESULTS, f"results/RESULTS.md no longer contains {original!r}"
        with pytest.raises(AssertionError, match=r"commits 79 community reports"):
            _check_the_model_drift_figure(
                RESULTS.replace(original, "all 78 demo community reports", 1)
            )

    def test_the_prescribed_attribution_satisfies_this_rule(self) -> None:
        """A red that cannot be answered is worse than no red at all, so the sentence
        the failure asks for is proven to satisfy the rule rather than promised to.

        Both accepted roads are shown: the two commits a reader can `git show`, and a
        committed record carrying the comparison.
        """
        original = (
            "That is the whole of the drift, and it is why the model layers are "
            "described as reproducing the protocol rather than the bytes."
        )
        assert original in RESULTS, (
            f"results/RESULTS.md no longer contains {original!r}; re-point this control "
            "at the sentence that closes the drift paragraph now"
        )
        for attribution in (
            "The two generations are committed at `52238c8` and `c607ff4`, so "
            "`git diff --name-only 52238c8 c607ff4 -- fixtures/summaries/demo` names the "
            "one report that moved.",
            "The comparison is carried in `results/summary-drift.json`.",
        ):
            assert _check_the_model_drift_figure(
                RESULTS.replace(original, f"{original} {attribution}", 1)
            )

    def test_an_unattributed_figure_fails(self) -> None:
        """The finding itself: a number no committed byte reproduces, published under a
        page-wide promise that every number is read from the record beside it."""
        stripped = re.sub(_COMMITISH, "that run", RESULTS)
        stripped = re.sub(r"results/[\w.-]+\.json", "the record", stripped)
        assert stripped != RESULTS, (
            "this control removed no attribution from results/RESULTS.md, so its red "
            "proves nothing; re-derive it from the page"
        )
        with pytest.raises(AssertionError, match=r"no committed byte can reproduce it"):
            _check_the_model_drift_figure(stripped)


class TestThePanelTableBindingsCanGoRed:
    """One control per bound figure, over the REAL document, mutated in memory.

    The first two rows are the evaluator's own, verbatim from round 5's report 3, Finding
    1: they passed 594 green tests. Each control asserts the string it moves is in the
    committed file (a control over a string that is not there proves nothing), then
    requires the checker to fail with the message that NAMES the figure — not merely to
    fail, which any unrelated breakage would satisfy.
    """

    PANEL_MUTATIONS: ClassVar[list[tuple[str, str, str]]] = [
        (
            "| verdicts | 0 SUPPORTS · 0 REFUTES · **20 NOT_ENOUGH_INFO** |",
            "| verdicts | 2 SUPPORTS · 0 REFUTES · **18 NOT_ENOUGH_INFO** |",
            r"the verdict row says 2 SUPPORTS",
        ),
        (
            "| verdicts carrying an evidence span | 6 of 20 |",
            "| verdicts carrying an evidence span | 9 of 20 |",
            r"panel table, grounded verdicts",
        ),
        (
            "| conflicts detected | 0 |",
            "| conflicts detected | 1 |",
            r"panel table, conflicts",
        ),
        (
            "| deterministic lens | 0 findings |",
            "| deterministic lens | 2 findings |",
            r"panel table, deterministic lens",
        ),
        (
            "| citations deleted by the hygiene pass | 0 on every reviewer of both runs |",
            "| citations deleted by the hygiene pass | 3 on every reviewer of both runs |",
            r"panel table, citations dropped",
        ),
        (
            "| whole findings discarded as malformed | 0 on every reviewer of both runs |",
            "| whole findings discarded as malformed | 4 on every reviewer of both runs |",
            r"panel table, malformed findings",
        ),
        (
            "| findings lost to unparseable output | 0: neither reviewer failed to parse |",
            "| findings lost to unparseable output | 1: neither reviewer failed to parse |",
            r"panel table, unparsed findings",
        ),
        (
            "| reviewers | 2, blind and parallel",
            "| reviewers | 3, blind and parallel",
            r"panel table, reviewers",
        ),
        (
            "| claims verified | 20 verdicts, every one judged twice",
            "| claims verified | 24 verdicts, every one judged twice",
            r"panel table, claims verified",
        ),
        (
            "| scores | methods 4/4/4 · novelty 4/4/**3**",
            "| scores | methods 4/4/4 · novelty 4/4/**1**",
            r"the scores cell says",
        ),
        (
            "| findings | 8 each",
            "| findings | 6 each",
            r"panel table, findings per reviewer",
        ),
    ]

    @pytest.mark.parametrize(("original", "mutated", "message"), PANEL_MUTATIONS)
    def test_moving_a_panel_table_cell_fails(
        self, original: str, mutated: str, message: str
    ) -> None:
        assert original in RESULTS, (
            f"results/RESULTS.md no longer contains {original!r}, so this control moves a "
            "row that is not there and proves nothing. Re-point it at the row as it reads "
            "now, or the figure has left the table and the binding above will say so."
        )
        with pytest.raises(AssertionError, match=message):
            _check_the_panel_table(RESULTS.replace(original, mutated, 1))

    def test_dropping_a_bound_row_from_the_panel_table_fails(self) -> None:
        """A figure does not stop being published by leaving the table."""
        original = "| conflicts detected | 0 |\n"
        assert original in RESULTS
        with pytest.raises(AssertionError, match=r"has no row for \['conflicts'\]"):
            _check_the_panel_table(RESULTS.replace(original, "", 1))

    def test_dropping_a_verdict_class_from_the_distribution_fails(self) -> None:
        """The mutation a rewrite makes without noticing: a zero class quietly deleted."""
        original = "| verdicts | 0 SUPPORTS · 0 REFUTES · **20 NOT_ENOUGH_INFO** |"
        assert original in RESULTS
        mutated = "| verdicts | 0 REFUTES · **20 NOT_ENOUGH_INFO** |"
        with pytest.raises(AssertionError, match=r"count for every verdict this code can emit"):
            _check_the_panel_table(RESULTS.replace(original, mutated, 1))

    SWAP_MUTATIONS: ClassVar[list[tuple[str, str, str]]] = [
        (
            "| 1 | 14 / 20 | 6 / 20 | swap-consistency 0.70 | 226,585 | 1,297.9 s |",
            "| 1 | 16 / 20 | 6 / 20 | swap-consistency 0.70 | 226,585 | 1,297.9 s |",
            r"swap table, run 1, consistent",
        ),
        (
            "| 2 | 14 / 20 | 6 / 20 | swap-consistency 0.70 | 226,585 | 1,296.7 s |",
            "| 2 | 14 / 20 | 4 / 20 | swap-consistency 0.70 | 226,585 | 1,296.7 s |",
            r"swap table, run 2, abstain",
        ),
        (
            "| 1 | 14 / 20 | 6 / 20 | swap-consistency 0.70 | 226,585 | 1,297.9 s |",
            "| 1 | 14 / 20 | 6 / 20 | swap-consistency 0.85 | 226,585 | 1,297.9 s |",
            r"swap table, run 1: the table says swap-consistency 0.85",
        ),
        (
            "| 1 | 14 / 20 | 6 / 20 | swap-consistency 0.70 | 226,585 | 1,297.9 s |",
            "| 1 | 14 / 20 | 6 / 20 | swap-consistency 0.70 | 326,585 | 1,297.9 s |",
            r"swap table, run 1, tokens",
        ),
        (
            "| 1 | 14 / 20 | 6 / 20 | swap-consistency 0.70 | 226,585 | 1,297.9 s |",
            "| 1 | 14 / 20 | 6 / 20 | swap-consistency 0.70 | 226,585 | 1,097.9 s |",
            r"swap table, run 1: the table says '1,097.9 s'",
        ),
    ]

    @pytest.mark.parametrize(("original", "mutated", "message"), SWAP_MUTATIONS)
    def test_moving_a_swap_table_cell_fails(
        self, original: str, mutated: str, message: str
    ) -> None:
        assert original in RESULTS, (
            f"results/RESULTS.md no longer contains {original!r}; re-point this control"
        )
        with pytest.raises(AssertionError, match=message):
            _check_the_swap_table(RESULTS.replace(original, mutated, 1))

    def test_dropping_a_run_from_the_swap_table_fails(self) -> None:
        original = "| 2 | 14 / 20 | 6 / 20 | swap-consistency 0.70 | 226,585 | 1,296.7 s |\n"
        assert original in RESULTS
        message = r"Every run this repository ships gets its own row"
        with pytest.raises(AssertionError, match=message):
            _check_the_swap_table(RESULTS.replace(original, "", 1))

    def test_moving_the_exclusion_figure_fails(self) -> None:
        """The evaluator's own: `26 chunks dropped` -> `46 chunks dropped`, in both files."""
        for where, original, mutated in (
            (
                "results/RESULTS.md",
                "26 chunks dropped, recorded in the record",
                "46 chunks dropped, recorded in the record",
            ),
            (
                "README.md",
                "(26 chunks dropped on each)",
                "(36 chunks dropped on each)",
            ),
        ):
            in_results = where.endswith("RESULTS.md")
            text = RESULTS if in_results else README
            assert original in text, (
                f"{where} no longer contains {original!r}; re-point this control"
            )
            mutation = text.replace(original, mutated, 1)
            results_text = mutation if in_results else RESULTS
            readme = README if in_results else mutation
            with pytest.raises(AssertionError, match=r"were dropped"):
                _check_the_exclusion_figures(results_text, readme)
