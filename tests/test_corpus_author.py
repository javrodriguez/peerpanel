"""Manifest authoring tests — ordering law, forced-include law, exclusion law."""

from __future__ import annotations

from pathlib import Path

import pytest

from peerpanel.corpus.author import (
    Candidate,
    author_manifest,
    build_candidates,
    refused_forced,
)
from peerpanel.corpus.models import CorpusDoc, FetchOutcome


def _fake_doc(pmcid: str, forced: bool) -> CorpusDoc:
    return CorpusDoc(
        pmcid=pmcid,
        version=1,
        md5="a" * 32,
        license_code="CC BY",
        title=f"T {pmcid}",
        authors=["A"],
        doi=None,
        citation="c",
        source_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        fetched_at="2026-08-31",
        forced_include=forced,
    )


def _wire_build_doc(monkeypatch: pytest.MonkeyPatch, refuse: set[str]) -> None:
    from peerpanel.corpus import author as author_mod

    def fake(pmcid: str, *, forced: bool = False, client: object = None) -> object:
        if pmcid in refuse:
            return FetchOutcome.LICENSE_REFUSED
        return (_fake_doc(pmcid, forced), f"text of {pmcid}".encode())

    monkeypatch.setattr(author_mod, "build_doc", fake)


class TestBuildCandidates:
    def test_order_and_dedupe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from peerpanel.corpus import author as author_mod

        monkeypatch.setattr(
            author_mod.eutils,
            "elink_pmc",
            lambda pmcid, link: ["PMC2", "PMC3"] if link == "pmc_pmc_cites" else ["PMC4"],
        )
        monkeypatch.setattr(author_mod.eutils, "esearch_pmc", lambda term, retmax: ["PMC3", "PMC5"])
        cands = build_candidates(["PMC1", "PMC2"], ["PMC1"], "term")
        assert [(c.pmcid, c.forced, c.origin) for c in cands] == [
            ("PMC1", True, "seed"),
            ("PMC2", True, "seed"),
            ("PMC3", False, "one_hop"),
            ("PMC4", False, "one_hop"),
            ("PMC5", False, "topical"),
        ]


class TestAuthorManifest:
    def test_stops_at_target_but_forced_always_attempted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _wire_build_doc(monkeypatch, refuse=set())
        cands = [
            Candidate("PMC1", False, "topical"),
            Candidate("PMC2", False, "topical"),
            Candidate("PMC3", True, "seed"),
        ]
        manifest, outcomes = author_manifest("demo", cands, target_size=1)
        assert [d.pmcid for d in manifest.docs] == ["PMC1", "PMC3"]
        assert outcomes["PMC3"] is FetchOutcome.FETCHED
        assert "PMC2" not in outcomes

    def test_refused_forced_is_surfaced_never_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wire_build_doc(monkeypatch, refuse={"PMC9"})
        cands = [Candidate("PMC9", True, "seed"), Candidate("PMC1", False, "topical")]
        manifest, outcomes = author_manifest("demo", cands, target_size=1)
        assert outcomes["PMC9"] is FetchOutcome.LICENSE_REFUSED
        assert refused_forced(cands, outcomes) == ["PMC9"]
        assert [d.pmcid for d in manifest.docs] == ["PMC1"]

    def test_excluded_pmcid_never_fetched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wire_build_doc(monkeypatch, refuse=set())
        cands = [Candidate("PMC7", True, "seed"), Candidate("PMC1", False, "topical")]
        manifest, outcomes = author_manifest("demo", cands, target_size=2, exclude_pmcids={"PMC7"})
        assert "PMC7" not in outcomes
        assert all(d.pmcid != "PMC7" for d in manifest.docs)

    def test_texts_written_when_dest_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _wire_build_doc(monkeypatch, refuse=set())
        cands = [Candidate("PMC1", True, "seed")]
        author_manifest("demo", cands, target_size=1, dest_texts=tmp_path)
        assert (tmp_path / "PMC1.txt").read_bytes() == b"text of PMC1"

    def test_forced_flag_recorded_on_doc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _wire_build_doc(monkeypatch, refuse=set())
        manifest, _ = author_manifest(
            "demo",
            [Candidate("PMC1", True, "seed"), Candidate("PMC2", False, "topical")],
            target_size=2,
        )
        assert [d.forced_include for d in manifest.docs] == [True, False]
