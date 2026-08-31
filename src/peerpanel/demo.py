"""The captured demo run: everything real, printed once, captured verbatim.

`make demo` tees this to `demo/raw-<utc>.log`. The curated transcript that
ships beside it is a strict SUBSET of that raw capture — a test enforces it,
because a hand-edited transcript would be a mock wearing a recording's label.

One command reproduces the whole thing from a clean clone; it is printed at
the end of the run and stated in the README.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from peerpanel.evals.ablation import render_table as render_ablation
from peerpanel.evals.ablation import run_ablation
from peerpanel.evals.planted_eval import render_table as render_planted
from peerpanel.evals.planted_eval import run_planted_eval
from peerpanel.orchestration import PanelProviders, run_panel

REPRODUCE = "make corpus && make demo-index && make demo-summaries && make demo"
DEMO_MANUSCRIPT = "met17-auxotroph"
PLANTED_MANUSCRIPT = "caprin-heterochromatin"


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def run_demo(root: Path, corpus: str = "demo") -> int:
    from peerpanel.providers import OllamaNativeChat, OllamaNativeEmbed, OllamaOpenAIChat

    started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    stats = json.loads((root / "artifacts" / corpus / "build_stats.json").read_text())
    manifest = json.loads((root / "corpus" / f"{corpus}.manifest.json").read_text())

    _rule("PeerPanel — captured demo run")
    print(f"started (UTC): {started}")
    print(f"reproduce with: {REPRODUCE}")
    print(
        f"\ncorpus: {len(manifest['docs'])} open-access documents · {stats['chunks']} chunks\n"
        f"graph:  {stats['nodes']} entities · {stats['edges']} edges · "
        f"{stats['communities_per_resolution']['1.0']} communities at resolution 1.0\n"
        f"index built by: {stats['provider']} · {stats['truncated_chunks']} truncated extractions"
    )

    _rule("1. Retrieval ablation — five rungs over the same queries")
    ablation = run_ablation(root, OllamaNativeEmbed(), corpus=corpus)
    print(render_ablation(ablation))

    _rule("2. Panel review of a real manuscript")
    providers = PanelProviders(
        methods=OllamaOpenAIChat("llama3.1:8b"),
        novelty=OllamaNativeChat("qwen2:7b"),
        verifier=OllamaOpenAIChat("llama3.1:8b"),
        converger=OllamaNativeChat("qwen2:7b"),
    )
    review = run_panel(
        root, root / "manuscripts" / f"{DEMO_MANUSCRIPT}.txt", providers, corpus=corpus
    )
    print(f"manuscript: {review.manuscript_doi}")
    print(f"excluded from this run: {review.excluded_docs} (its own published twin)")
    for output in review.reviewer_outputs:
        print(
            f"\n[{output.reviewer} · {output.model}] scores {output.scores} "
            f"confidence {output.confidence}"
        )
        for finding in output.findings:
            print(f"  - ({finding.dimension}/{finding.severity}) {finding.text}")
    supported = sum(1 for v in review.verdicts if v.verdict == "SUPPORTS")
    refuted = sum(1 for v in review.verdicts if v.verdict == "REFUTES")
    nei = sum(1 for v in review.verdicts if v.verdict == "NOT_ENOUGH_INFO")
    print(
        f"\nclaims: {supported} supported · {refuted} refuted · {nei} abstained · "
        f"swap-consistency {review.swap_consistency_rate} over n={review.n_swap_checked}"
    )
    print(f"deterministic lens: {len(review.deterministic_findings)} findings")
    for det in review.deterministic_findings:
        print(f"  - [{det.check}] {det.detail} ({det.location})")
    print(f"conflicts: {len(review.conflicts)}")
    for conflict in review.conflicts:
        print(f"  - {conflict.type}: {conflict.detail}")
    print(f"\nmeta-review:\n{review.summary}")
    print(f"\npanel cost: {review.total_tokens} tokens · {review.wall_s}s")

    _rule("3. Planted-error evaluation — panel vs equal-compute single agent")
    planted = run_planted_eval(
        root,
        root / "manuscripts" / f"{PLANTED_MANUSCRIPT}.txt",
        providers,
        baseline_provider=OllamaOpenAIChat("llama3.1:8b"),
        corpus=corpus,
    )
    print(render_planted(planted))
    out = root / "artifacts" / corpus / "planted_eval.json"
    out.write_text(planted.model_dump_json(indent=1) + "\n")

    _rule("end of captured run")
    print(f"finished (UTC): {dt.datetime.now(dt.UTC).isoformat(timespec='seconds')}")
    print(f"reproduce with: {REPRODUCE}")
    return 0
