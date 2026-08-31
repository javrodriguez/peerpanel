"""Author corpus/demo.manifest.json — the T2.1 sizing duty, from measured cost.

Scope (DECISIONS D8): two query manuscripts (MET17 + oleaginous-biopolymer;
caprin deferred to the >=3 expansion). Forced = their gated-in citation seeds
+ their two twins; topical top-up to TARGET_TOTAL from the committed ordered
candidates. The caprin twin is left out entirely (its manuscript is not in
the run set). Texts land in corpus/demo/ (gitignored), pins in the manifest.

Run: uv run --directory <workspace> python scripts/author_demo_manifest.py
"""

from __future__ import annotations

import json
from pathlib import Path

from peerpanel.corpus.author import Candidate, author_manifest, refused_forced
from peerpanel.corpus.citations import render
from peerpanel.corpus.models import CorpusManifest

ROOT = Path(__file__).resolve().parents[1]
TARGET_TOTAL = 68
RUN_MANUSCRIPTS = ("10.1101/2023.05.18.541364", "10.1101/2025.05.04.652101")
RUN_TWINS = ("PMC10729969", "PMC12628781")
DEFERRED_TWIN = "PMC11918387"  # caprin — its manuscript is deferred


def main() -> None:
    ground_truth = json.loads((ROOT / "corpus" / "ground_truth.json").read_text())["manuscripts"]
    seeds: list[str] = []
    for doi in RUN_MANUSCRIPTS:
        for pmcid in ground_truth[doi]["cited_pmcids"]:
            if pmcid not in seeds:
                seeds.append(pmcid)
    candidates = [Candidate(p, True, "seed") for p in seeds]
    candidates += [Candidate(p, True, "twin") for p in RUN_TWINS]
    ordered = json.loads((ROOT / "corpus" / "demo.candidates.json").read_text())
    known = {c.pmcid for c in candidates} | {DEFERRED_TWIN}
    candidates += [
        Candidate(str(row["pmcid"]), False, "topical")
        for row in ordered
        if row["origin"] == "topical" and str(row["pmcid"]) not in known
    ]
    manifest, outcomes = author_manifest(
        "demo",
        candidates,
        target_size=TARGET_TOTAL,
        exclude_pmcids={DEFERRED_TWIN},
        dest_texts=ROOT / "corpus" / "demo",
    )
    manifest.save(ROOT / "corpus" / "demo.manifest.json")
    citations = render(manifest)
    (ROOT / "corpus" / "DEMO_CITATIONS.md").write_text(citations)
    refused = refused_forced(candidates, outcomes)
    gated_in = {
        doi: sorted(
            set(ground_truth[doi]["cited_pmcids"]) & manifest.pmcids()
        )
        for doi in RUN_MANUSCRIPTS
    }
    summary = {
        "docs": len(manifest.docs),
        "forced_in": sum(1 for d in manifest.docs if d.forced_include),
        "refused_forced": refused,
        "gt_in_corpus": {doi: len(v) for doi, v in gated_in.items()},
        "gt_aggregate": sum(len(v) for v in gated_in.values()),
    }
    (ROOT / "artifacts" / "demo_manifest_summary.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=1, sort_keys=True))
    loaded = CorpusManifest.load(ROOT / "corpus" / "demo.manifest.json")
    assert len(loaded.docs) == len(manifest.docs)
    print("manifest verified reloadable")


if __name__ == "__main__":
    main()
