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
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

from peerpanel.corpus.models import CorpusManifest
from peerpanel.manuscripts.store import read_manuscript

ROOT = Path(__file__).resolve().parents[1]
RESULTS = (ROOT / "results" / "RESULTS.md").read_text()
README = (ROOT / "README.md").read_text()

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
