"""Fixture-pipeline tests: deterministic chunking of the committed CI corpus,
the committed fixture's two no-model truths (payload hash + chunk-id
alignment), CLI drift semantics, and the live regeneration road."""

from __future__ import annotations

from pathlib import Path

import pytest

from peerpanel.__main__ import main
from peerpanel.embeddings import pipeline

ROOT = Path(__file__).resolve().parents[1]


class TestCiChunks:
    def test_deterministic_across_passes(self) -> None:
        first, _ = pipeline.ci_chunks(ROOT)
        second, _ = pipeline.ci_chunks(ROOT)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert [c.text for c in first] == [c.text for c in second]

    def test_covers_every_manifest_doc(self) -> None:
        from peerpanel.corpus.models import CorpusManifest

        manifest = CorpusManifest.load(ROOT / pipeline.CI_MANIFEST)
        chunks, _ = pipeline.ci_chunks(ROOT)
        assert {c.doc_id for c in chunks} == manifest.pmcids()

    def test_sanitizer_ran_not_silenced(self) -> None:
        """Findings are reported, never dropped — the corpus may legitimately
        trip zero screens, but the pipeline must return the findings channel."""
        _, findings = pipeline.ci_chunks(ROOT)
        assert isinstance(findings, list)


class TestCommittedFixture:
    def test_stored_fixture_matches_hash_and_current_chunks(self) -> None:
        """The CI-honest deterministic check: committed vectors match their
        sha256 manifest AND their chunk ids match a fresh sanitize+chunk pass
        over the committed corpus. Drift in corpus text, sanitizer, chunker or
        fixture bytes fails here, with no model in the loop."""
        assert pipeline.check_stored(ROOT)


class TestCliSemantics:
    def test_check_exit_codes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(pipeline, "check_live", lambda root, provider: (True, "abc", "abc"))
        assert main(["embeddings", "--check"]) == 0
        monkeypatch.setattr(pipeline, "check_live", lambda root, provider: (False, "abc", "def"))
        assert main(["embeddings", "--check"]) == 1


@pytest.mark.ollama
class TestLiveRegeneration:
    def test_recomputed_hash_matches_committed(self) -> None:
        from peerpanel.providers import OllamaNativeEmbed

        match, recomputed, committed = pipeline.check_live(ROOT, OllamaNativeEmbed())
        assert match, f"recomputed {recomputed} != committed {committed}"
