"""Ingest one query manuscript from bioRxiv: fetch, extract, attribute, record.

The road: the bioRxiv details API (`https://api.biorxiv.org/details/biorxiv/<doi>`)
names a JATS XML URL per version; the pinned version's XML is fetched, gated on
the API-reported license (cc_by / cc0 only — nothing else is redistributable),
extracted to deterministic plain text, and written as `<slug>.txt` with a full
attribution header. The same run records the preprint->published twin in
`twins.json` (the self-exclusion predicate's data) and regenerates
`CITATIONS.md` from the committed files. Everything on disk is reproducible by
the exact command each file's header carries.

Run it as: `uv run python -m peerpanel.manuscripts.ingest <doi> --version N
--slug S --published-doi D --pmcid P`
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from peerpanel.corpus.s3 import LicenseRefused

from .jats import extract
from .store import ManuscriptHeader, read_manuscript, write_manuscript
from .twins import Twin, TwinTable, load_twins

DETAILS_API = "https://api.biorxiv.org/details/biorxiv/{doi}"
USER_AGENT = "peerpanel/0.1 (open-source research tool; JATS fetch for attribution)"
LICENSE_TEXT = "CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/"
CHANGES_NOTE = (
    "converted from JATS XML to plain text; figures, tables, display math and the "
    "reference list stripped; section headings kept as their own paragraphs"
)
TWINS_FILENAME = "twins.json"
_REDISTRIBUTABLE = frozenset({"cc_by", "cc0"})
_RETRY_STATUSES = frozenset({429, 503})
_MAX_ATTEMPTS = 5


@dataclass(frozen=True)
class BiorxivVersion:
    """One pinned version of a bioRxiv preprint, per the details API."""

    doi: str
    version: int
    license: str
    jatsxml_url: str


def fetch_version_meta(
    doi: str, version: int, client: httpx.Client | None = None
) -> BiorxivVersion:
    """Resolve the pinned version's metadata; errors if the version is absent."""
    c = client or httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT})
    resp = c.get(DETAILS_API.format(doi=doi))
    resp.raise_for_status()
    entries: list[dict[str, Any]] = resp.json().get("collection", [])
    for entry in entries:
        if entry.get("doi") == doi and int(entry.get("version", -1)) == version:
            return BiorxivVersion(
                doi=doi,
                version=version,
                license=str(entry.get("license") or ""),
                jatsxml_url=str(entry["jatsxml"]),
            )
    have = sorted(int(e.get("version", -1)) for e in entries)
    raise ValueError(f"{doi}: version {version} not in the details API (has {have})")


def fetch_jats(meta: BiorxivVersion, client: httpx.Client | None = None) -> bytes:
    """Fetch the pinned version's JATS bytes, backing off politely on 429/503."""
    c = client or httpx.Client(
        timeout=60.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    )
    for attempt in range(_MAX_ATTEMPTS):
        resp = c.get(meta.jatsxml_url)
        if resp.status_code in _RETRY_STATUSES and attempt < _MAX_ATTEMPTS - 1:
            time.sleep(45.0 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.content
    raise RuntimeError("unreachable")  # pragma: no cover — the loop returns or raises


def regenerate_command(doi: str, version: int, slug: str, published_doi: str, pmcid: str) -> str:
    return (
        f"uv run python -m peerpanel.manuscripts.ingest {doi} --version {version} "
        f"--slug {slug} --published-doi {published_doi} --pmcid {pmcid}"
    )


CITATIONS_HEADER = """\
# Manuscript citations

The query manuscripts under review. Each remains under its own license, held
by its authors — the repo's MIT license covers code only. Each text was
converted from the bioRxiv JATS XML to plain text (figures, tables, display
math and the reference list stripped); every file's own attribution header
pins the source JATS bytes by md5 and carries the exact command that
regenerates it. The published-twin line records the identity `twins.json`
uses to exclude a manuscript's published form from retrieval while that
manuscript is under review.
"""


def render_citations(out_dir: Path, table: TwinTable) -> str:
    """CITATIONS.md, derived purely from the committed files + the twin table."""
    lines = [CITATIONS_HEADER]
    for path in sorted(out_dir.glob("*.txt")):
        header, _ = read_manuscript(path)
        authors = "; ".join(header.authors)
        twin = next((t for t in table.twins if t.preprint_doi == header.preprint_doi), None)
        twin_line = (
            f"  **Published twin:** DOI {twin.published_doi} · {twin.pmcid} "
            "(excluded from retrieval while under review)\n"
            if twin
            else ""
        )
        lines.append(
            f"- **Title:** {header.title}\n"
            f"  **Authors:** {authors}\n"
            f"  **Source:** {header.source} (DOI: {header.preprint_doi})\n"
            f"  **License:** {header.license}\n"
            f"  **Changes:** {header.changes}\n"
            f"{twin_line}"
            f"  **Pinned:** {header.preprint_doi} v{header.preprint_version} "
            f"· JATS md5 {header.source_jats_md5}\n"
        )
    return "\n".join(lines)


def ingest(
    doi: str,
    version: int,
    slug: str,
    published_doi: str,
    pmcid: str,
    out_dir: Path,
    client: httpx.Client | None = None,
) -> Path:
    """The whole road for one manuscript; returns the committed .txt path."""
    meta = fetch_version_meta(doi, version, client)
    if meta.license not in _REDISTRIBUTABLE:
        raise LicenseRefused(
            f"{doi} v{version}: bioRxiv license {meta.license!r} is not redistributable "
            f"(need one of {sorted(_REDISTRIBUTABLE)})"
        )
    jats_bytes = fetch_jats(meta, client)
    digest = hashlib.md5(jats_bytes).hexdigest()
    extracted = extract(jats_bytes)

    header = ManuscriptHeader(
        title=extracted.title,
        authors=extracted.authors,
        source=f"https://www.biorxiv.org/content/{doi}v{version}",
        license=LICENSE_TEXT,
        changes=CHANGES_NOTE,
        preprint_doi=doi,
        preprint_version=version,
        source_jats_md5=digest,
        regenerate=regenerate_command(doi, version, slug, published_doi, pmcid),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{slug}.txt"
    write_manuscript(path, header, extracted.body)

    twins_path = out_dir / TWINS_FILENAME
    table = load_twins(twins_path) if twins_path.exists() else TwinTable(twins=[])
    table.upsert(
        Twin(
            preprint_doi=doi,
            preprint_version=version,
            published_doi=published_doi,
            pmcid=pmcid,
            title=extracted.title,
        )
    )
    table.save(twins_path)
    (out_dir / "CITATIONS.md").write_text(
        render_citations(out_dir, table), encoding="utf-8", newline="\n"
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m peerpanel.manuscripts.ingest",
        description="Fetch + extract one query manuscript from bioRxiv (pinned version).",
    )
    parser.add_argument("doi", help="preprint DOI, e.g. 10.1101/2023.05.18.541364")
    parser.add_argument("--version", type=int, required=True, help="pinned bioRxiv version")
    parser.add_argument("--slug", required=True, help="output filename stem")
    parser.add_argument("--published-doi", required=True, help="the published twin's DOI")
    parser.add_argument("--pmcid", required=True, help="the published twin's PMCID")
    parser.add_argument("--out", type=Path, default=Path("manuscripts"), help="output dir")
    args = parser.parse_args(argv)
    path = ingest(args.doi, args.version, args.slug, args.published_doi, args.pmcid, args.out)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
