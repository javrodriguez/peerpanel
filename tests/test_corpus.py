"""Corpus layer tests: the license/integrity gates, the sync status grid, and
attribution rendering — deterministic here; the live roads carry the network marker."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from peerpanel.corpus import ALLOWED_LICENSES, CorpusDoc, CorpusManifest, FetchOutcome
from peerpanel.corpus.citations import render
from peerpanel.corpus.s3 import IntegrityError, LicenseRefused, S3Meta, _md5_from_url, gate
from peerpanel.corpus.sync import doc_path, sync, verify


def _meta(**overrides: object) -> S3Meta:
    base: dict[str, object] = {
        "pmcid": "PMC1",
        "version": 1,
        "license_code": "CC BY",
        "is_retracted": False,
        "is_pmc_openaccess": True,
        "title": "t",
        "doi": "10.1/x",
        "citation": "c",
        "text_md5": "0" * 32,
    }
    base.update(overrides)
    return S3Meta(**base)  # type: ignore[arg-type]


def _doc(pmcid: str, md5: str, **overrides: object) -> CorpusDoc:
    base: dict[str, object] = {
        "pmcid": pmcid,
        "version": 1,
        "md5": md5,
        "license_code": "CC BY",
        "title": "T",
        "authors": ["A One"],
        "doi": "10.1/y",
        "citation": "J. 2024",
        "source_url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        "fetched_at": "2026-08-30",
    }
    base.update(overrides)
    return CorpusDoc(**base)  # type: ignore[arg-type]


class TestGate:
    def test_cc_by_and_cc0_pass(self) -> None:
        gate(_meta(license_code="CC BY"), ALLOWED_LICENSES)
        gate(_meta(license_code="CC0"), ALLOWED_LICENSES)

    @pytest.mark.parametrize("bad", ["CC BY-NC", "CC BY-ND", "CC BY-SA", "", "unknown"])
    def test_non_redistributable_licenses_refused(self, bad: str) -> None:
        with pytest.raises(LicenseRefused):
            gate(_meta(license_code=bad), ALLOWED_LICENSES)

    def test_retracted_refused_even_with_good_license(self) -> None:
        with pytest.raises(LicenseRefused, match="retracted"):
            gate(_meta(is_retracted=True), ALLOWED_LICENSES)

    def test_non_oa_refused(self) -> None:
        with pytest.raises(LicenseRefused, match="open-access"):
            gate(_meta(is_pmc_openaccess=False), ALLOWED_LICENSES)


class TestMd5Parsing:
    def test_extracts_pinned_md5(self) -> None:
        url = "s3://pmc-oa-opendata/PMC1.1/PMC1.1.txt?md5=89b749cfa818ce9c540d7c01e0880c0f"
        assert _md5_from_url(url) == "89b749cfa818ce9c540d7c01e0880c0f"

    def test_missing_md5_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            _md5_from_url("s3://bucket/key.txt")


class TestManifest:
    def test_roundtrip(self, tmp_path: Path) -> None:
        m = CorpusManifest(name="ci", docs=[_doc("PMC1", "a" * 32)])
        p = tmp_path / "m.json"
        m.save(p)
        assert CorpusManifest.load(p) == m

    def test_citations_render_carries_tasl_and_pin(self) -> None:
        text = render(CorpusManifest(name="ci", docs=[_doc("PMC1", "a" * 32)]))
        for needle in ("**Title:** T", "**Authors:** A One", "**License:** CC BY",
                       "creativecommons.org/licenses/by/4.0", "**Pinned:** PMC1.1 · md5"):
            assert needle in text
        assert "MIT license covers code only" in text


class TestSyncGrid:
    """The status-by-operation grid, driven through monkeypatched fetch seams
    (unit stubs of our own network layer — the live roads are network-marked)."""

    def _wire(self, monkeypatch: pytest.MonkeyPatch, body: bytes, meta: S3Meta) -> None:
        from peerpanel.corpus import sync as sync_mod

        monkeypatch.setattr(sync_mod.s3, "fetch_meta", lambda p, v, c=None: meta)
        monkeypatch.setattr(sync_mod.s3, "fetch_text", lambda m, c=None: body)

    def test_absent_then_fetched_then_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = b"real text"
        md5 = hashlib.md5(body).hexdigest()
        manifest = CorpusManifest(name="ci", docs=[_doc("PMC1", md5)])
        self._wire(monkeypatch, body, _meta(pmcid="PMC1", text_md5=md5))
        assert sync(manifest, tmp_path)["PMC1"] is FetchOutcome.FETCHED
        assert doc_path(tmp_path, "PMC1").read_bytes() == body
        assert sync(manifest, tmp_path)["PMC1"] is FetchOutcome.CACHED
        assert verify(manifest, tmp_path) == {"PMC1": True}

    def test_upstream_drift_refused_as_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = CorpusManifest(name="ci", docs=[_doc("PMC1", "a" * 32)])
        self._wire(monkeypatch, b"x", _meta(pmcid="PMC1", text_md5="b" * 32))
        assert sync(manifest, tmp_path)["PMC1"] is FetchOutcome.MD5_MISMATCH
        assert not doc_path(tmp_path, "PMC1").exists()

    def test_license_flip_refused_on_refetch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = CorpusManifest(name="ci", docs=[_doc("PMC1", "a" * 32)])
        meta = _meta(pmcid="PMC1", license_code="CC BY-NC", text_md5="a" * 32)
        self._wire(monkeypatch, b"x", meta)
        assert sync(manifest, tmp_path)["PMC1"] is FetchOutcome.LICENSE_REFUSED

    def test_retraction_refused_on_refetch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manifest = CorpusManifest(name="ci", docs=[_doc("PMC1", "a" * 32)])
        self._wire(monkeypatch, b"x", _meta(pmcid="PMC1", is_retracted=True, text_md5="a" * 32))
        assert sync(manifest, tmp_path)["PMC1"] is FetchOutcome.RETRACTED_REFUSED

    def test_cold_start_twin_fresh_dir_equals_cached_rerun(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        body = b"same bytes"
        md5 = hashlib.md5(body).hexdigest()
        manifest = CorpusManifest(name="ci", docs=[_doc("PMC1", md5)])
        self._wire(monkeypatch, body, _meta(pmcid="PMC1", text_md5=md5))
        fresh = tmp_path / "fresh"
        sync(manifest, fresh)
        rerun = sync(manifest, fresh)
        assert rerun["PMC1"] is FetchOutcome.CACHED
        assert verify(manifest, fresh)["PMC1"]


@pytest.mark.network
class TestLiveRoads:
    def test_s3_meta_gate_and_md5_verified_fetch(self) -> None:
        from peerpanel.corpus import s3 as s3_mod

        version = s3_mod.resolve_latest_version("PMC10729969")
        assert version is not None
        meta = s3_mod.fetch_meta("PMC10729969", version)
        gate(meta, ALLOWED_LICENSES)
        body = s3_mod.fetch_text(meta)
        assert len(body) > 100_000
        assert hashlib.md5(body).hexdigest() == meta.text_md5

    def test_md5_tamper_detection(self) -> None:
        from dataclasses import replace

        from peerpanel.corpus import s3 as s3_mod

        version = s3_mod.resolve_latest_version("PMC10729969")
        assert version is not None
        meta = replace(s3_mod.fetch_meta("PMC10729969", version), text_md5="0" * 32)
        with pytest.raises(IntegrityError):
            s3_mod.fetch_text(meta)

    def test_discovery_and_link_expansion(self) -> None:
        from peerpanel.corpus import eutils

        hits = eutils.esearch_pmc(
            f'"Saccharomyces cerevisiae"[TIAB] AND methionine[TIAB] AND {eutils.CC_FILTER}',
            retmax=5,
        )
        assert len(hits) == 5
        assert all(h.startswith("PMC") for h in hits)
        cited_by = eutils.elink_pmc("PMC10729969", "pmc_pmc_citedby")
        assert isinstance(cited_by, list)
        authors = eutils.esummary_authors(["PMC10729969"])
        assert authors["PMC10729969"]
