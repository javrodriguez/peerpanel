"""The recorded preprint->published twin table — the self-exclusion predicate.

This recorded table is the sole sound exclusion predicate for the
self-exclusion invariant: a preprint has no PMCID of its own, so naive
DOI/PMCID matching between a manuscript under review and the corpus excludes
NOTHING — its published twin sits in the corpus under a different DOI and its
own PMCID. Only this table, recorded at ingest time from the verified
publisher links, can say which PMCIDs are the same work. At review and eval
time every retrieval index must EXCLUDE `excluded_pmcids(preprint_doi)` for
the manuscript under review, and an unknown DOI raises rather than returning
an empty set (a silently-empty exclusion would defeat the invariant unseen).
The planted-error held-out papers join this same table at C4, so one
predicate covers both roads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel


class Twin(BaseModel):
    """One manuscript's identity pair: the preprint and its published form."""

    preprint_doi: str
    preprint_version: int
    published_doi: str
    pmcid: str
    title: str


@dataclass
class TwinTable:
    twins: list[Twin]

    def excluded_pmcids(self, preprint_doi: str) -> set[str]:
        """The PMCIDs every retrieval index must exclude while this preprint
        is under review. Raises KeyError for a DOI the table has never seen."""
        matches = {t.pmcid for t in self.twins if t.preprint_doi == preprint_doi}
        if not matches:
            raise KeyError(
                f"{preprint_doi}: not in the twin table — refusing to answer with an "
                "empty exclusion set; record the twin before reviewing this manuscript"
            )
        return matches

    def upsert(self, twin: Twin) -> None:
        """Add or replace the entry for `twin.preprint_doi` (one entry per preprint)."""
        self.twins = [t for t in self.twins if t.preprint_doi != twin.preprint_doi]
        self.twins.append(twin)
        self.twins.sort(key=lambda t: t.preprint_doi)

    def save(self, path: Path) -> None:
        data = [t.model_dump() for t in sorted(self.twins, key=lambda t: t.preprint_doi)]
        path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def load_twins(path: Path) -> TwinTable:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TwinTable(twins=[Twin.model_validate(item) for item in raw])
