"""Corpus data models: the pinned manifest and per-document records."""

from __future__ import annotations

import enum
import json
from pathlib import Path

from pydantic import BaseModel

ALLOWED_LICENSES = frozenset({"CC BY", "CC0"})


class CorpusDoc(BaseModel):
    """One pinned document: identity, integrity, license, attribution."""

    pmcid: str
    version: int
    md5: str
    license_code: str
    title: str
    authors: list[str]
    doi: str | None
    citation: str
    source_url: str
    fetched_at: str
    forced_include: bool = False


class CorpusManifest(BaseModel):
    """The pinned corpus: (PMCID, version, md5) triples plus attribution data."""

    name: str
    docs: list[CorpusDoc]

    @classmethod
    def load(cls, path: Path) -> CorpusManifest:
        return cls.model_validate_json(path.read_text())

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.model_dump(), indent=1, sort_keys=True) + "\n")

    def pmcids(self) -> set[str]:
        return {d.pmcid for d in self.docs}


class FetchOutcome(enum.Enum):
    """The status grid a sync can land in, per document."""

    FETCHED = "fetched"
    CACHED = "cached"
    MD5_MISMATCH = "md5_mismatch"
    LICENSE_REFUSED = "license_refused"
    RETRACTED_REFUSED = "retracted_refused"
    MISSING_UPSTREAM = "missing_upstream"
