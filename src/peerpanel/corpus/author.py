"""Manifest authoring: citation-seeded corpus construction.

Order is load-bearing (plan T1.2/T1.3): the query manuscripts' cited papers
come FIRST (forced — truncation may never drop them), then one-hop citation
neighbours, then topical top-up. Every candidate passes the authoritative S3
gate; a refused FORCED candidate is surfaced in the outcomes, never silently
dropped, because forced includes are the eval's ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import eutils
from .models import CorpusManifest, FetchOutcome
from .sync import build_doc, doc_path


@dataclass(frozen=True)
class Candidate:
    pmcid: str
    forced: bool
    origin: str  # "seed" | "one_hop" | "topical"


def build_candidates(
    seed_pmcids: list[str],
    one_hop_from: list[str],
    topical_term: str,
    *,
    topical_retmax: int = 200,
) -> list[Candidate]:
    """Ordered, deduplicated candidate list: seeds -> one-hop -> topical."""
    seen: set[str] = set()
    out: list[Candidate] = []

    def add(pmcid: str, forced: bool, origin: str) -> None:
        if pmcid not in seen:
            seen.add(pmcid)
            out.append(Candidate(pmcid, forced, origin))

    for pmcid in seed_pmcids:
        add(pmcid, True, "seed")
    for source in one_hop_from:
        for link in ("pmc_pmc_cites", "pmc_pmc_citedby"):
            for pmcid in eutils.elink_pmc(source, link):
                add(pmcid, False, "one_hop")
    for pmcid in eutils.esearch_pmc(topical_term, retmax=topical_retmax):
        add(pmcid, False, "topical")
    return out


def author_manifest(
    name: str,
    candidates: list[Candidate],
    target_size: int,
    *,
    exclude_pmcids: set[str] | None = None,
    dest_texts: Path | None = None,
) -> tuple[CorpusManifest, dict[str, FetchOutcome]]:
    """Walk candidates in order until target_size docs pass the gate.

    Forced candidates are ALWAYS attempted (even past target_size — they are
    ground truth); non-forced ones stop once the target is met. exclude_pmcids
    supports held-out designs (a planted-error paper must not enter its own
    retrieval corpus).
    """
    excluded = exclude_pmcids or set()
    manifest = CorpusManifest(name=name, docs=[])
    outcomes: dict[str, FetchOutcome] = {}
    for cand in candidates:
        if cand.pmcid in excluded:
            continue
        accepted = len(manifest.docs)
        if not cand.forced and accepted >= target_size:
            continue
        result = build_doc(cand.pmcid, forced=cand.forced)
        if isinstance(result, FetchOutcome):
            outcomes[cand.pmcid] = result
            continue
        doc, body = result
        outcomes[cand.pmcid] = FetchOutcome.FETCHED
        manifest.docs.append(doc)
        if dest_texts is not None:
            dest_texts.mkdir(parents=True, exist_ok=True)
            doc_path(dest_texts, doc.pmcid).write_bytes(body)
    return manifest, outcomes


def refused_forced(candidates: list[Candidate], outcomes: dict[str, FetchOutcome]) -> list[str]:
    """The forced candidates that did NOT make it in — must be reported, they
    shrink the eval's ground truth."""
    return [
        c.pmcid
        for c in candidates
        if c.forced and outcomes.get(c.pmcid) not in (FetchOutcome.FETCHED, None)
    ]
