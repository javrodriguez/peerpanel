"""The captured demo: the whole arc, narrated, on the demo-scale corpus.

This is tier 2 — it runs the real panel against real models, so it needs
Ollama. Its value is as EVIDENCE: the raw capture is committed, and the one
command that reproduces it is printed in the capture itself.

The narration exists because a reader should be able to follow what happened
without reading the JSON: what the corpus is, what the index looks like, what
the manuscript's own twin exclusion actually removed, what each reviewer said,
and where the panel abstained rather than guessed.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from peerpanel.corpus.models import CorpusManifest
from peerpanel.graph.pipeline import chunks_for
from peerpanel.graph.run_graph import build_run_graph
from peerpanel.manuscripts.store import read_manuscript
from peerpanel.manuscripts.twins import load_twins
from peerpanel.orchestration import PanelProviders, run_panel

REPRODUCE = "make corpus && make demo"
MANUSCRIPT = "met17-auxotroph"


def _rule(title: str) -> str:
    return f"\n{'─' * 78}\n{title}\n{'─' * 78}"


def run_demo(root: Path, providers: PanelProviders, corpus: str = "demo") -> int:
    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    manifest = CorpusManifest.load(root / "corpus" / f"{corpus}.manifest.json")
    manuscript_path = root / "manuscripts" / f"{MANUSCRIPT}.txt"
    header, _body = read_manuscript(manuscript_path)
    excluded = load_twins(root / "manuscripts" / "twins.json").excluded_pmcids(
        header.preprint_doi
    )

    print(f"PeerPanel demo · captured {started}")
    print(f"reproduce this exactly with: {REPRODUCE}")

    print(_rule("1. The corpus — pinned, licensed, verifiable"))
    licenses = sorted({d.license_code for d in manifest.docs})
    print(f"  {len(manifest.docs)} documents · licenses {licenses} · every one md5-pinned")
    forced = sum(1 for d in manifest.docs if d.forced_include)
    print(f"  forced (citation-seeded ground truth): {forced}")
    print("  attribution for each: corpus/DEMO_CITATIONS.md")

    print(_rule("2. The manuscript under review, and what that excludes"))
    print(f"  {header.title}")
    print(f"  preprint DOI {header.preprint_doi} ({header.license})")
    print(f"  its published twin is in this corpus: {sorted(excluded)}")
    print("  so the twin is withheld — from the index AND from the graph, which is")
    print("  rebuilt without its extractions. A manuscript cannot retrieve itself.")

    print(_rule("3. The knowledge graph, built and then rebuilt for this run"))
    chunks = chunks_for(root, corpus)
    full, _fa, _fw = build_run_graph(root, corpus, set())
    run, _ra, withheld = build_run_graph(root, corpus, excluded)
    print(f"  {len(chunks)} chunks over {len(manifest.docs)} documents")
    print(
        f"  full corpus graph:  {full.number_of_nodes():>6} entities · "
        f"{full.number_of_edges():>6} edges"
    )
    print(
        f"  this run's graph:   {run.number_of_nodes():>6} entities · "
        f"{run.number_of_edges():>6} edges"
    )
    print(f"  withheld for this run: {withheld} chunks, "
          f"{full.number_of_nodes() - run.number_of_nodes()} entities, "
          f"{full.number_of_edges() - run.number_of_edges()} edges")

    print(_rule("4. The panel — blind, parallel, structurally different lenses"))
    print("  running (this is real model work; expect several minutes)…")
    review = run_panel(root, manuscript_path, providers, corpus=corpus)

    for output in review.reviewer_outputs:
        print(f"\n  [{output.reviewer}] via {output.model}")
        print(f"    scores {output.scores} · confidence {output.confidence}")
        for finding in output.findings[:4]:
            grounding = (
                f" (grounded in {len(finding.evidence_chunk_ids)} retrieved chunks)"
                if finding.evidence_chunk_ids else " (about the manuscript text itself)"
            )
            print(f"    · {finding.severity}/{finding.dimension}: {finding.text[:150]}{grounding}")

    print(_rule("5. Adversarial claim verification — order-swapped, abstaining"))
    supports = [v for v in review.verdicts if v.verdict == "SUPPORTS"]
    refutes = [v for v in review.verdicts if v.verdict == "REFUTES"]
    nei = [v for v in review.verdicts if v.verdict == "NOT_ENOUGH_INFO"]
    inconsistent = [v for v in review.verdicts if not v.swap_consistent]
    print(f"  {len(review.verdicts)} claims judged twice each (evidence order reversed)")
    print(f"  SUPPORTS {len(supports)} · REFUTES {len(refutes)} · NOT_ENOUGH_INFO {len(nei)}")
    print(f"  order-inconsistent (forced to abstain): {len(inconsistent)}")
    print(f"  swap-consistency rate: {review.swap_consistency_rate} over n={review.n_swap_checked}")
    for verdict in (supports + refutes)[:3]:
        print(f"    · {verdict.verdict}: {verdict.claim_text[:120]}")
        for span in verdict.evidence[:1]:
            print(f"        evidence {span.chunk_id}: \"{span.quote[:100]}\"")

    print(_rule("6. The deterministic lens — no model involved"))
    if review.deterministic_findings:
        for check in review.deterministic_findings:
            print(
                f"  · {check.check} ({check.severity}) at {check.location}: "
                f"{check.detail}"
            )
    else:
        print("  no findings — this is a published, peer-reviewed manuscript, so the")
        print("  gene-symbol, reference-integrity and statistics checks all pass. The")
        print("  lens is proven on planted defects instead: make eval.")

    print(_rule("7. Convergence"))
    if review.conflicts:
        for conflict in review.conflicts:
            print(f"  {conflict.type} conflict between {conflict.between}: {conflict.detail}")
    else:
        print("  no reviewer conflicts detected this run")
    print(f"\n  meta-review:\n    {review.summary[:900]}")

    print(_rule("Cost of this run"))
    print(f"  {review.total_tokens} tokens · {review.wall_s}s wall-clock · $0 (local models)")
    out = root / "artifacts" / corpus / f"demo-review-{MANUSCRIPT}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(review.model_dump_json(indent=1) + "\n")
    print(f"  structured record: {out.relative_to(root)}")
    print(f"\nreproduce: {REPRODUCE}")
    return 0


def replay(root: Path, corpus: str = "demo") -> int:
    """Print the committed capture rather than re-running it."""
    path = root / "results" / "demo-capture.log"
    if not path.exists():
        print("no committed capture yet — run `make demo` (needs Ollama).")
        return 3
    print(path.read_text())
    return 0
