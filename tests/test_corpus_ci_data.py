"""Pins on the COMMITTED corpus data — pure local, no network.

These tests are what make the committed corpus trustworthy: every byte matches
its manifest pin, every document is redistributable, every document is
attributed, and the demo candidate list satisfies the ground-truth gate input.
"""

from __future__ import annotations

import json
from pathlib import Path

from peerpanel.corpus import ALLOWED_LICENSES, CorpusManifest
from peerpanel.corpus.sync import verify

ROOT = Path(__file__).resolve().parents[1]
TWINS = {"PMC10729969", "PMC12628781", "PMC11918387"}
SHOWCASE = "PMC10729969"


def _manifest() -> CorpusManifest:
    return CorpusManifest.load(ROOT / "corpus" / "ci.manifest.json")


class TestCiCorpusPins:
    def test_every_committed_byte_matches_its_pin(self) -> None:
        results = verify(_manifest(), ROOT / "corpus" / "ci")
        assert results and all(results.values()), {k: v for k, v in results.items() if not v}

    def test_every_license_is_redistributable(self) -> None:
        assert all(d.license_code in ALLOWED_LICENSES for d in _manifest().docs)

    def test_showcase_pin_present_forced_and_alone(self) -> None:
        forced = [d.pmcid for d in _manifest().docs if d.forced_include]
        assert forced == [SHOWCASE]

    def test_every_doc_attributed_in_citations_md(self) -> None:
        citations = (ROOT / "CITATIONS.md").read_text()
        for doc in _manifest().docs:
            assert f"**Pinned:** {doc.pmcid}.{doc.version} · md5 {doc.md5}" in citations

    def test_every_doc_has_authors_and_title(self) -> None:
        for doc in _manifest().docs:
            assert doc.authors, doc.pmcid
            assert doc.title, doc.pmcid


class TestDemoCandidates:
    def _candidates(self) -> list[dict[str, object]]:
        data = json.loads((ROOT / "corpus" / "demo.candidates.json").read_text())
        assert isinstance(data, list)
        return data

    def test_ground_truth_gate_input_satisfied(self) -> None:
        forced = [c for c in self._candidates() if c["forced"]]
        assert len(forced) >= 20, "the eval ground-truth gate needs >=20 citation seeds"

    def test_seeds_come_first_and_are_forced(self) -> None:
        cands = self._candidates()
        first_non_forced = next(i for i, c in enumerate(cands) if not c["forced"])
        assert all(c["forced"] for c in cands[:first_non_forced])
        assert all(c["origin"] in ("seed", "twin") for c in cands if c["forced"])

    def test_all_three_twins_are_forced_members(self) -> None:
        """Twins must be IN the demo corpus so each manuscript's run has its own
        twin in-index to drop — an absent twin makes the per-run exclusion set
        empty, the vacuous state the self-exclusion law forbids."""
        twins = {str(c["pmcid"]): c for c in self._candidates() if c["origin"] == "twin"}
        assert set(twins) == TWINS
        assert all(c["forced"] for c in twins.values())


class TestDemoManifest:
    """Pins on the committed demo manifest (D8 scope: MET17 + biopolymer)."""

    RUN_TWINS = {"PMC10729969", "PMC12628781"}
    DEFERRED_TWIN = "PMC11918387"

    def _manifest(self) -> CorpusManifest:
        return CorpusManifest.load(ROOT / "corpus" / "demo.manifest.json")

    def test_scope_and_ground_truth_gate(self) -> None:
        import json

        manifest = self._manifest()
        assert len(manifest.docs) == 68
        ground_truth = json.loads((ROOT / "corpus" / "ground_truth.json").read_text())
        aggregate = 0
        for doi in ("10.1101/2023.05.18.541364", "10.1101/2025.05.04.652101"):
            cited = set(ground_truth["manuscripts"][doi]["cited_pmcids"])
            aggregate += len(cited & manifest.pmcids())
        assert aggregate >= 20, "the demo corpus must clear the ground-truth gate"

    def test_run_twins_in_deferred_twin_out(self) -> None:
        """Per D8: exclusion binds per RUN; caprin has no run, so its twin is
        not a member — it re-enters WITH its manuscript at the expansion."""
        pmcids = self._manifest().pmcids()
        assert self.RUN_TWINS <= pmcids
        assert self.DEFERRED_TWIN not in pmcids

    def test_every_doc_redistributable_and_attributed(self) -> None:
        manifest = self._manifest()
        assert all(d.license_code in ALLOWED_LICENSES for d in manifest.docs)
        citations = (ROOT / "corpus" / "DEMO_CITATIONS.md").read_text()
        for doc in manifest.docs:
            assert f"**Pinned:** {doc.pmcid}.{doc.version} · md5 {doc.md5}" in citations
