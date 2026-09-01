"""Rungs must be compared over lists of the same length.

A comparison is only a comparison if the things compared are alike. This one was
not: retrieval depth was set in CHUNKS while scoring ran over DOCUMENTS, so the
document lists actually judged differed by a factor of three between rungs while
every reported cell carried the same `@k` label. The rung with the longest list
looked best at depth, and the rung with the shortest was reported against a
ceiling it could not reach.

Prose cannot hold this. The condition is asserted here so it stays true whoever
edits the harness next.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _report(corpus: str) -> dict[str, object]:
    return json.loads((ROOT / "results" / f"ablation-{corpus}.json").read_text())


class TestEqualDocumentDepth:
    @pytest.mark.parametrize("corpus", ["ci", "demo"])
    def test_every_rung_ranks_the_same_number_of_documents(self, corpus: str) -> None:
        rows = _report(corpus)["results"]
        by_case: dict[str, dict[str, int]] = defaultdict(dict)
        for row in rows:  # type: ignore[union-attr]
            by_case[str(row["case_doi"])][str(row["rung"])] = int(row["docs_ranked"])
        for case, depths in by_case.items():
            assert len(set(depths.values())) == 1, (
                f"{case}: rungs were scored over different-length document lists "
                f"{depths} — the cells are not comparable"
            )

    @pytest.mark.parametrize("corpus", ["ci", "demo"])
    def test_the_scored_depth_is_recorded_not_implied(self, corpus: str) -> None:
        """Both depths ship in the artifact: the chunk depth each rung needed, and
        the document depth actually judged. A reader should never have to infer
        either from a label."""
        for row in _report(corpus)["results"]:  # type: ignore[union-attr]
            assert int(row["chunks_retrieved"]) > 0
            assert "docs_ranked" in row

    def test_rungs_needed_different_chunk_depths_to_get_there(self) -> None:
        """Proves the equalisation is doing work rather than being a no-op: if
        every rung reached the target at the same chunk depth, the old fixed-chunk
        approach would have been fair by accident and this test would be idle."""
        depths = {
            (str(r["rung"]), str(r["case_doi"])): int(r["chunks_retrieved"])
            for r in _report("demo")["results"]  # type: ignore[union-attr]
        }
        assert len(set(depths.values())) > 1, (
            "all rungs reached the target document depth at the same chunk depth — "
            "check this test is still meaningful"
        )
