"""Eval-harness tests: metric math, the N gate, and real committed-data cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from peerpanel.evals import (
    N_GATE,
    QueryCase,
    build_cases,
    doc_ranking,
    ndcg_at_k,
    per_item_hits,
    rates_allowed,
    recall_at_k,
)
from peerpanel.manuscripts.twins import load_twins
from peerpanel.retrieval.base import RetrievalHit

ROOT = Path(__file__).resolve().parents[1]
TWINS = load_twins(ROOT / "manuscripts" / "twins.json").twins


def _hits(*chunk_ids: str) -> list[RetrievalHit]:
    return [RetrievalHit(cid, 1.0 / (i + 1), i + 1) for i, cid in enumerate(chunk_ids)]


class TestMetricMath:
    def test_doc_ranking_dedupes_in_order(self) -> None:
        ranking = doc_ranking(_hits("a:0:x", "b:0:x", "a:1:y", "c:0:x"))
        assert ranking == ["a", "b", "c"]

    def test_recall(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"a", "c", "d"}, k=3) == pytest.approx(2 / 3)
        assert recall_at_k(["a"], set(), k=3) == 0.0

    def test_ndcg_perfect_and_worst(self) -> None:
        assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)
        assert ndcg_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0
        early = ndcg_at_k(["a", "x", "y"], {"a"}, k=3)
        late = ndcg_at_k(["x", "y", "a"], {"a"}, k=3)
        assert early > late

    def test_per_item_hits_reports_misses(self) -> None:
        rows = per_item_hits(["a", "b"], {"a", "z"}, k=5)
        assert rows == [("a", 1), ("z", None)]


class TestGate:
    def _case(self, n: int) -> QueryCase:
        return QueryCase(
            preprint_doi="10.1/x",
            manuscript_file="m.txt",
            query="q",
            relevant_docs=[f"PMC{i}" for i in range(n)],
            excluded_docs=["PMC0000"],
        )

    def test_gate_binds_at_aggregate_twenty(self) -> None:
        assert not rates_allowed([self._case(10), self._case(9)])
        assert rates_allowed([self._case(10), self._case(10)])
        assert N_GATE == 20


class TestRealCases:
    """Built from the COMMITTED data — pins the ground-truth wiring end to end."""

    def test_cases_build_from_committed_data(self) -> None:
        cases = build_cases(ROOT, Path("corpus") / "ci.manifest.json")
        assert len(cases) == 3
        by_doi = {c.preprint_doi: c for c in cases}
        met17 = by_doi["10.1101/2023.05.18.541364"]
        # The twin is corpus/ci's showcase pin: it must be named as excluded and it
        # must not appear among the documents the case counts as relevant.
        assert "PMC10729969" in met17.excluded_docs
        assert "PMC10729969" not in met17.relevant_docs
        for case in cases:
            # `case.excluded_docs` cannot be empty — excluded_pmcids() RAISES on an
            # unknown DOI rather than returning an empty set — so asserting it is
            # non-empty proves nothing. What is falsifiable, and what matters, is
            # that the exclusion names this manuscript's own twin and no other's.
            twin = {t.pmcid for t in TWINS if t.preprint_doi == case.preprint_doi}
            assert set(case.excluded_docs) == twin
            assert not set(case.excluded_docs) & set(case.relevant_docs)
            assert case.query.startswith(
                ("When is an auxotroph", "Fission yeast", "Acidification by nitrogen")
            )

    def test_ci_corpus_rates_honestly_gated(self) -> None:
        cases = build_cases(ROOT, Path("corpus") / "ci.manifest.json")
        aggregate = sum(len(c.relevant_docs) for c in cases)
        # The CI corpus is topical+showcase by design (D5) — citation seeds live
        # in the DEMO corpus. Whatever the number is, the gate must tell the truth.
        assert rates_allowed(cases) == (aggregate >= N_GATE)
