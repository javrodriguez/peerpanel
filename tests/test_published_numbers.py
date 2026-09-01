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
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = (ROOT / "results" / "RESULTS.md").read_text()
README = (ROOT / "README.md").read_text()

ROW_LABELS = {
    "BM25": "bm25",
    "vector": "vector",
    "RRF hybrid": "rrf-hybrid",
    "GraphRAG local": "graphrag-local",
    "GraphRAG global": "graphrag-global",
}


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
        match = re.search(
            rf"^\| \**{re.escape(label)}\** \|(.+)$", RESULTS, re.MULTILINE
        )
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
            at_k = sum(
                1 for r in rows for _, rank in r["hits"] if rank is not None and rank <= k
            )
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
        """Any 'Nx slower/faster' claim must match the artifact it summarises."""
        data = _ablation()

        def mean(rung: str) -> float:
            rows = data[rung]
            return sum(float(r["latency_ms"]) for r in rows) / len(rows)  # type: ignore[arg-type]

        real = mean("graphrag-local") / mean("bm25")
        for text, where in ((RESULTS, "RESULTS.md"), (README, "README.md")):
            for claimed in re.findall(r"(\d+(?:\.\d+)?)\s*[x\u00d7]\s*(?:slower|faster)", text):
                assert float(claimed) == pytest.approx(real, abs=1.0), (
                    f"{where} claims {claimed}x; the artifact gives {real:.1f}x"
                )


class TestPanelGlossesMatchTheTable:
    """The plain-English sentences about swap-consistency, against the runs."""

    def _flipped(self) -> list[tuple[int, int]]:
        out = []
        for name in ("panel-review-met17.json", "panel-review-met17-run2.json"):
            data = json.loads((ROOT / "results" / name).read_text())
            flipped = sum(1 for v in data["verdicts"] if not v["swap_consistent"])
            out.append((flipped, int(data["n_swap_checked"])))
        return out

    def test_per_twelve_and_per_ten_glosses(self) -> None:
        rates = [f / n for f, n in self._flipped()]
        lo, hi = min(rates), max(rates)
        for unit, phrase in ((12, "of every twelve"), (10, "in every ten")):
            for match in re.finditer(
                rf"(\w+) (?:and|to) (\w+) {re.escape(phrase)}", RESULTS + README
            ):
                words = {"two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7}
                claimed_lo = words.get(match.group(1).lower())
                claimed_hi = words.get(match.group(2).lower())
                if claimed_lo is None or claimed_hi is None:
                    continue
                assert claimed_lo == pytest.approx(lo * unit, abs=0.5), (
                    f"'{match.group(0)}' understates: measured {lo * unit:.1f} per {unit}"
                )
                assert claimed_hi == pytest.approx(hi * unit, abs=0.5), (
                    f"'{match.group(0)}' overstates: measured {hi * unit:.1f} per {unit}"
                )

    def test_swap_rates_quoted_match_the_records(self) -> None:
        published = {float(m) for m in re.findall(r"swap-consistency (\d\.\d+)", RESULTS + README)}
        actual = set()
        for name in ("panel-review-met17.json", "panel-review-met17-run2.json"):
            data = json.loads((ROOT / "results" / name).read_text())
            actual.add(round(float(data["swap_consistency_rate"]), 2))
        for value in published:
            assert round(value, 2) in actual, f"{value} is not either run's rate: {actual}"


class TestIndexTableMatchesBuildStats:
    def test_entity_and_edge_counts(self) -> None:
        for corpus, column in (("ci", 1), ("demo", 2)):
            stats = json.loads((ROOT / "results" / f"build-stats-{corpus}.json").read_text())
            for field, label in (("nodes", "entities"), ("edges", "edges")):
                match = re.search(rf"^\| {label} \|(.+)$", RESULTS, re.MULTILINE)
                assert match, f"no {label} row"
                cells = [c.strip() for c in match.group(1).split("|") if c.strip()]
                assert _cell(cells[column - 1]) == stats[field], f"{corpus} {label}"


class TestPlantedEvalMatchesItsArtifact:
    """The newest published numbers, and the ones a reader weighs most."""

    def _report(self) -> dict[str, object]:
        return json.loads((ROOT / "results" / "planted-eval-demo.json").read_text())

    def _arm(self, name: str) -> dict[str, object]:
        return next(r for r in self._report()["results"] if r["system"] == name)  # type: ignore[index,arg-type]

    def test_detection_counts(self) -> None:
        report = self._report()
        panel, base = self._arm("panel"), self._arm("single-agent-equal-compute")
        n = int(report["n_errors"])  # type: ignore[call-overload]
        for text in (RESULTS, README):
            assert f"**{len(panel['detected'])} / {n}**" in text  # type: ignore[arg-type]
            assert f"**{len(base['detected'])} / {n}**" in text  # type: ignore[arg-type]

    def test_token_and_wall_figures(self) -> None:
        panel, base = self._arm("panel"), self._arm("single-agent-equal-compute")
        for text in (RESULTS, README):
            assert f"{int(panel['total_tokens']):,}" in text  # type: ignore[call-overload]
            assert f"{int(base['total_tokens']):,}" in text  # type: ignore[call-overload]

    def test_the_multiplier_claims(self) -> None:
        """'4.5x the tokens and 8x the wall-clock' must match the artifact."""
        panel, base = self._arm("panel"), self._arm("single-agent-equal-compute")
        token_ratio = float(panel["total_tokens"]) / float(base["total_tokens"])  # type: ignore[arg-type]
        wall_ratio = float(panel["wall_s"]) / float(base["wall_s"])  # type: ignore[arg-type]
        for text, where in ((RESULTS, "RESULTS.md"), (README, "README.md")):
            claimed_tokens = re.search(r"(\d+(?:\.\d+)?)[x\u00d7] the tokens", text)
            claimed_wall = re.search(r"(\d+(?:\.\d+)?)[x\u00d7] the wall-clock", text)
            if claimed_tokens:
                assert float(claimed_tokens.group(1)) == pytest.approx(token_ratio, abs=0.2), (
                    f"{where}: claims {claimed_tokens.group(1)}x tokens; "
                    f"artifact {token_ratio:.2f}x"
                )
            if claimed_wall:
                assert float(claimed_wall.group(1)) == pytest.approx(wall_ratio, abs=0.5), (
                    f"{where}: claims {claimed_wall.group(1)}x wall; artifact {wall_ratio:.2f}x"
                )

    def test_the_budget_share_claim(self) -> None:
        """'22% of the panel's compute' — the caveat that keeps the loss honest."""
        panel, base = self._arm("panel"), self._arm("single-agent-equal-compute")
        share = float(base["total_tokens"]) / float(panel["total_tokens"]) * 100  # type: ignore[arg-type]
        for match in re.finditer(r"(\d+)% of the panel", RESULTS + README):
            assert float(match.group(1)) == pytest.approx(share, abs=2), (
                f"claims {match.group(1)}%; artifact gives {share:.0f}%"
            )
