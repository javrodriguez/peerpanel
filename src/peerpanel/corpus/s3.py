"""PMC Open Access content fetch: the public pmc-oa-opendata S3 bucket over
plain anonymous HTTPS (the OA Web Service oa.fcgi was retired Feb 2026).

This is the AUTHORITATIVE license gate: a document enters the corpus only if
its own S3 metadata says CC BY / CC0, is not retracted, and its text bytes
match the md5 the metadata pins.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import httpx

BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com"
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


class LicenseRefused(Exception):
    """The authoritative gate said no (license, retraction, or OA status)."""


class IntegrityError(Exception):
    """Downloaded bytes do not match the pinned md5."""


@dataclass(frozen=True)
class S3Meta:
    pmcid: str
    version: int
    license_code: str
    is_retracted: bool
    is_pmc_openaccess: bool
    title: str
    doi: str | None
    citation: str
    text_md5: str


def resolve_latest_version(pmcid: str, client: httpx.Client | None = None) -> int | None:
    """List the bucket by prefix and return the highest version, None if absent."""
    c = client or httpx.Client(timeout=30.0)
    resp = c.get(BUCKET, params={"list-type": "2", "prefix": f"{pmcid}."})
    resp.raise_for_status()
    versions: set[int] = set()
    for key_el in ET.fromstring(resp.text).iter(f"{_S3_NS}Key"):
        m = re.match(rf"{pmcid}\.(\d+)/", key_el.text or "")
        if m:
            versions.add(int(m.group(1)))
    return max(versions) if versions else None


def _md5_from_url(url: str) -> str:
    m = re.search(r"[?&]md5=([0-9a-f]{32})", url)
    if not m:
        raise ValueError(f"no md5 in metadata url: {url}")
    return m.group(1)


def fetch_meta(pmcid: str, version: int, client: httpx.Client | None = None) -> S3Meta:
    c = client or httpx.Client(timeout=30.0)
    resp = c.get(f"{BUCKET}/{pmcid}.{version}/{pmcid}.{version}.json")
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    return S3Meta(
        pmcid=data["pmcid"],
        version=int(data["version"]),
        license_code=str(data.get("license_code") or ""),
        is_retracted=bool(data.get("is_retracted")),
        is_pmc_openaccess=bool(data.get("is_pmc_openaccess")),
        title=str(data.get("title") or ""),
        doi=data.get("doi"),
        citation=str(data.get("citation") or ""),
        text_md5=_md5_from_url(str(data["text_url"])),
    )


def gate(meta: S3Meta, allowed_licenses: frozenset[str]) -> None:
    """The authoritative two-of-three gate; raises with the reason."""
    if meta.license_code not in allowed_licenses:
        raise LicenseRefused(
            f"{meta.pmcid}: license {meta.license_code!r} not in {sorted(allowed_licenses)}"
        )
    if meta.is_retracted:
        raise LicenseRefused(f"{meta.pmcid}: retracted — never enters the corpus")
    if not meta.is_pmc_openaccess:
        raise LicenseRefused(f"{meta.pmcid}: not in the PMC open-access subset")


def fetch_text(meta: S3Meta, client: httpx.Client | None = None) -> bytes:
    """Fetch the plain-text body and verify it against the pinned md5."""
    c = client or httpx.Client(timeout=60.0)
    resp = c.get(f"{BUCKET}/{meta.pmcid}.{meta.version}/{meta.pmcid}.{meta.version}.txt")
    resp.raise_for_status()
    body = resp.content
    digest = hashlib.md5(body).hexdigest()
    if digest != meta.text_md5:
        raise IntegrityError(f"{meta.pmcid}: md5 {digest} != pinned {meta.text_md5}")
    return body
