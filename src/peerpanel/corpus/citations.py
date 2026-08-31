"""CITATIONS.md generation — TASL attribution (Title, Author, Source, License)
plus the changes-made note CC BY 4.0 requires for the XML->text extraction."""

from __future__ import annotations

from .models import CorpusManifest

_LICENSE_URLS = {
    "CC BY": "https://creativecommons.org/licenses/by/4.0/",
    "CC0": "https://creativecommons.org/publicdomain/zero/1.0/",
}

HEADER = """\
# Corpus citations

Every document below remains under its own license, held by its authors — the
repo's MIT license covers code only. Text was extracted by PMC from the
publisher XML to plain text (figures and markup stripped); we redistribute
those extractions unmodified, pinned by md5.
"""


def render(manifest: CorpusManifest) -> str:
    lines = [HEADER]
    for doc in sorted(manifest.docs, key=lambda d: d.pmcid):
        authors = "; ".join(doc.authors) if doc.authors else "(authors per citation below)"
        license_url = _LICENSE_URLS.get(doc.license_code, "")
        doi = f" (DOI: {doc.doi})" if doc.doi else ""
        lines.append(
            f"- **Title:** {doc.title}\n"
            f"  **Authors:** {authors}\n"
            f"  **Source:** {doc.source_url}{doi}\n"
            f"  **Citation:** {doc.citation}\n"
            f"  **License:** {doc.license_code} — {license_url}\n"
            f"  **Changes:** publisher XML extracted to plain text by PMC; redistributed as-is.\n"
            f"  **Pinned:** {doc.pmcid}.{doc.version} · md5 {doc.md5}\n"
        )
    return "\n".join(lines)
