"""NCBI E-utilities: discovery only.

PMC prohibits bulk content retrieval through E-utilities — discovery (esearch),
link expansion (elink) and summaries (esummary) happen here; document CONTENT
comes from the PMC Open Access S3 bucket (s3.py). Calls are throttled to stay
inside NCBI's no-key guidance (<=3 req/s).
"""

from __future__ import annotations

import time
from typing import Any

import httpx

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "peerpanel"
_MIN_INTERVAL_S = 0.34
_last_call = 0.0

CC_FILTER = 'open access[filter] AND ("cc by license"[filter] OR "cc0 license"[filter])'


def _throttled_get(url: str, params: dict[str, Any], timeout: float = 30.0) -> httpx.Response:
    global _last_call
    wait = _MIN_INTERVAL_S - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    resp = httpx.get(url, params={**params, "tool": TOOL}, timeout=timeout)
    _last_call = time.monotonic()
    resp.raise_for_status()
    return resp


def esearch_pmc(term: str, retmax: int = 200) -> list[str]:
    """Search PMC; returns PMCIDs (PMC-prefixed). Caller supplies license filters
    via CC_FILTER — this is the first, non-authoritative license gate."""
    resp = _throttled_get(
        f"{EUTILS}/esearch.fcgi",
        {"db": "pmc", "retmode": "json", "retmax": retmax, "term": term},
    )
    ids: list[str] = resp.json()["esearchresult"]["idlist"]
    return [f"PMC{i}" for i in ids]


def _strip(pmcid: str) -> str:
    return pmcid.removeprefix("PMC")


def elink_pmc(pmcid: str, linkname: str) -> list[str]:
    """Follow one elink relation from a PMC article (e.g. pmc_pmc_cites,
    pmc_pmc_citedby); returns PMCIDs."""
    resp = _throttled_get(
        f"{EUTILS}/elink.fcgi",
        {
            "dbfrom": "pmc",
            "db": "pmc",
            "retmode": "json",
            "id": _strip(pmcid),
            "linkname": linkname,
        },
    )
    out: list[str] = []
    for linkset in resp.json().get("linksets", []):
        for db in linkset.get("linksetdbs", []):
            if db.get("linkname") == linkname:
                out.extend(f"PMC{i}" for i in db.get("links", []))
    return out


def esummary_authors(pmcids: list[str]) -> dict[str, list[str]]:
    """Author lists for a batch of PMCIDs (the S3 metadata JSON carries none)."""
    if not pmcids:
        return {}
    resp = _throttled_get(
        f"{EUTILS}/esummary.fcgi",
        {"db": "pmc", "retmode": "json", "id": ",".join(_strip(p) for p in pmcids)},
    )
    result = resp.json().get("result", {})
    out: dict[str, list[str]] = {}
    for uid in result.get("uids", []):
        entry = result.get(uid, {})
        out[f"PMC{uid}"] = [a["name"] for a in entry.get("authors", []) if a.get("name")]
    return out
