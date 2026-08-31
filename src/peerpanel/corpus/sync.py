"""Manifest-driven corpus sync: fetch, verify, refuse — the full status grid.

Every (state x operation) cell lands in an explicit FetchOutcome:
absent -> FETCHED · present+md5-ok -> CACHED · present+md5-drift -> re-fetch,
MD5_MISMATCH if upstream disagrees with the pin · gate failure ->
LICENSE_REFUSED / RETRACTED_REFUSED · gone upstream -> MISSING_UPSTREAM.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import httpx

from . import eutils, s3
from .models import ALLOWED_LICENSES, CorpusDoc, CorpusManifest, FetchOutcome


def doc_path(dest: Path, pmcid: str) -> Path:
    return dest / f"{pmcid}.txt"


def _refusal(e: s3.LicenseRefused) -> FetchOutcome:
    return FetchOutcome.RETRACTED_REFUSED if "retracted" in str(e) else FetchOutcome.LICENSE_REFUSED


def _local_md5(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_doc(
    pmcid: str, *, forced: bool = False, client: httpx.Client | None = None
) -> tuple[CorpusDoc, bytes] | FetchOutcome:
    """Resolve, gate, fetch and pin one document (used when authoring a manifest)."""
    version = s3.resolve_latest_version(pmcid, client)
    if version is None:
        return FetchOutcome.MISSING_UPSTREAM
    meta = s3.fetch_meta(pmcid, version, client)
    try:
        s3.gate(meta, ALLOWED_LICENSES)
    except s3.LicenseRefused as e:
        return _refusal(e)
    body = s3.fetch_text(meta, client)
    authors = eutils.esummary_authors([pmcid]).get(pmcid, [])
    doc = CorpusDoc(
        pmcid=pmcid,
        version=version,
        md5=meta.text_md5,
        license_code=meta.license_code,
        title=meta.title,
        authors=authors,
        doi=meta.doi,
        citation=meta.citation,
        source_url=f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/",
        fetched_at=dt.date.today().isoformat(),
        forced_include=forced,
    )
    return doc, body


def sync(
    manifest: CorpusManifest, dest: Path, client: httpx.Client | None = None
) -> dict[str, FetchOutcome]:
    """Bring dest to the manifest's pinned state; every doc gets an outcome."""
    dest.mkdir(parents=True, exist_ok=True)
    outcomes: dict[str, FetchOutcome] = {}
    for doc in manifest.docs:
        path = doc_path(dest, doc.pmcid)
        local = _local_md5(path)
        if local == doc.md5:
            outcomes[doc.pmcid] = FetchOutcome.CACHED
            continue
        meta = s3.fetch_meta(doc.pmcid, doc.version, client)
        try:
            s3.gate(meta, ALLOWED_LICENSES)
        except s3.LicenseRefused as e:
            outcomes[doc.pmcid] = _refusal(e)
            continue
        if meta.text_md5 != doc.md5:
            # Upstream no longer matches the pin: refuse rather than silently drift.
            outcomes[doc.pmcid] = FetchOutcome.MD5_MISMATCH
            continue
        body = s3.fetch_text(meta, client)
        path.write_bytes(body)
        outcomes[doc.pmcid] = FetchOutcome.FETCHED
    return outcomes


def verify(manifest: CorpusManifest, dest: Path) -> dict[str, bool]:
    """Pure local check: does every on-disk byte match its pin?"""
    return {d.pmcid: _local_md5(doc_path(dest, d.pmcid)) == d.md5 for d in manifest.docs}
