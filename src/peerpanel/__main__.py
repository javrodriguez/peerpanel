"""PeerPanel CLI. Commands land checkpoint by checkpoint; an unbuilt command says so honestly."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PLANNED = ("quickstart", "demo", "corpus", "embeddings", "graph", "ablation", "eval")


def _corpus(args: list[str]) -> int:
    from peerpanel.corpus.models import CorpusManifest
    from peerpanel.corpus.sync import sync, verify

    root = Path.cwd()
    sub = args[0] if args else "fetch"
    if sub == "fetch":
        manifest_path = root / "corpus" / "demo.manifest.json"
        if not manifest_path.exists():
            print(
                "corpus fetch: no corpus/demo.manifest.json yet — it is written at "
                "checkpoint C2 (T2.1). The committed CI corpus needs no fetch.",
                file=sys.stderr,
            )
            return 3
        manifest = CorpusManifest.load(manifest_path)
        outcomes = sync(manifest, root / "corpus" / "demo")
        counts = Counter(o.value for o in outcomes.values())
        print(f"corpus fetch: {dict(counts)} of {len(manifest.docs)} docs")
        bad = {p: o.value for p, o in outcomes.items() if o.value not in ("fetched", "cached")}
        if bad:
            print(f"corpus fetch: NOT satisfied: {bad}", file=sys.stderr)
            return 1
        return 0
    if sub == "verify":
        manifest = CorpusManifest.load(root / "corpus" / "ci.manifest.json")
        results = verify(manifest, root / "corpus" / "ci")
        ok = all(results.values())
        failed = sorted(p for p, good in results.items() if not good)
        print(f"corpus verify: {sum(results.values())}/{len(results)} md5-exact")
        if not ok:
            print(f"corpus verify: FAILED: {failed}", file=sys.stderr)
        return 0 if ok else 1
    print(f"corpus: unknown subcommand {sub!r} (fetch | verify)", file=sys.stderr)
    return 2


def _embeddings(args: list[str]) -> int:
    from peerpanel.embeddings import pipeline
    from peerpanel.providers import OllamaNativeEmbed

    root = Path.cwd()
    if "--write" in args:
        digest = pipeline.write(root, OllamaNativeEmbed())
        print(f"embeddings: fixture written, payload sha256 {digest}")
        return 0
    # Default (and --check): recompute live, diff against the committed manifest.
    match, recomputed, committed = pipeline.check_live(root, OllamaNativeEmbed())
    if match:
        print(f"embeddings: unchanged — recomputed sha256 matches committed {committed}")
        return 0
    print(
        f"embeddings: DRIFT — recomputed {recomputed} != committed {committed}; "
        "inspect, then accept deliberately with `embeddings --write`",
        file=sys.stderr,
    )
    return 1


def _corpus_arg(args: list[str]) -> str:
    return args[args.index("--corpus") + 1] if "--corpus" in args else "ci"


def _graph(args: list[str]) -> int:
    root = Path.cwd()
    sub = args[0] if args else "build"
    corpus = _corpus_arg(args)
    if sub == "build":
        from peerpanel.graph.pipeline import graph_build, write_demo_embedding_fixture
        from peerpanel.providers import OllamaNativeEmbed, OllamaOpenAIChat

        if corpus == "demo":
            digest = write_demo_embedding_fixture(root, OllamaNativeEmbed())
            print(f"demo embedding fixture written, sha256 {digest}")
        result = graph_build(root, OllamaOpenAIChat("llama3.1:8b"), corpus=corpus)
        print(json.dumps(result, indent=1, sort_keys=True))
        truncated = result.get("truncated_chunks")
        if isinstance(truncated, int) and truncated:
            print(f"graph build: {truncated} chunk(s) recorded as truncated", file=sys.stderr)
        return 0
    if sub == "summaries":
        from peerpanel.graph.build import load_graph
        from peerpanel.graph.summaries import summarise_communities
        from peerpanel.providers import OllamaOpenAIChat

        graph = load_graph(root / "artifacts" / corpus / "graph.json")
        communities = json.loads((root / "artifacts" / corpus / "communities.json").read_text())
        provider = OllamaOpenAIChat("llama3.1:8b")
        cache = root / "fixtures" / "summaries" / corpus
        # --resolution limits which Leiden levels get LLM reports. The hierarchy is
        # always detected at every level (communities.json); reports cost one model
        # call each, so by default only the level retrieval actually reads is
        # summarised. Stated in the README rather than implied.
        wanted = (
            {args[args.index("--resolution") + 1]}
            if "--resolution" in args
            else set(communities)
        )
        all_reports: list[dict[str, object]] = []
        for resolution, assignment in communities.items():
            if resolution not in wanted:
                continue
            typed = {n: int(c) for n, c in assignment.items()}
            reports = summarise_communities(
                graph, typed, float(resolution), provider, cache_dir=cache
            )
            all_reports.extend(r.model_dump() for r in reports)
            print(f"resolution {resolution}: {len(reports)} community reports")
        out = root / "artifacts" / corpus / "summaries.json"
        out.write_text(json.dumps(all_reports, indent=1, sort_keys=True) + "\n")
        truncated = sum(1 for r in all_reports if bool(r.get("truncated")))
        if truncated:
            print(f"graph summaries: {truncated} report(s) truncated", file=sys.stderr)
        print(f"graph summaries: {len(all_reports)} reports -> {out}")
        return 0
    print(f"graph: unknown subcommand {sub!r} (build | summaries)", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print(f"peerpanel commands: {' '.join(PLANNED)}")
        return 0
    command, rest = args[0], args[1:]
    if command == "quickstart":
        from peerpanel.quickstart import run_quickstart

        return run_quickstart(Path.cwd())
    if command == "corpus":
        return _corpus(rest)
    if command == "embeddings":
        return _embeddings(rest)
    if command == "graph":
        return _graph(rest)
    if command == "ablation":
        from peerpanel.evals.ablation import render_table, run_ablation
        from peerpanel.providers import OllamaNativeEmbed

        corpus = _corpus_arg(rest)
        report = run_ablation(Path.cwd(), OllamaNativeEmbed(), corpus=corpus)
        out = Path.cwd() / "artifacts" / corpus / "ablation.json"
        out.write_text(report.model_dump_json(indent=1) + "\n")
        print(render_table(report))
        print(f"ablation: full report -> {out}")
        return 0
    if command == "review":
        from peerpanel.orchestration import PanelProviders, run_panel
        from peerpanel.providers import OllamaNativeChat, OllamaOpenAIChat

        names = [a for a in rest if not a.startswith("--")]
        if not names:
            print("review: name a manuscript (a file under manuscripts/)", file=sys.stderr)
            return 2
        manuscript = Path.cwd() / "manuscripts" / f"{Path(names[0]).stem}.txt"
        if not manuscript.exists():
            print(f"review: no manuscript at {manuscript}", file=sys.stderr)
            return 2
        corpus = _corpus_arg(rest) if "--corpus" in rest else "demo"
        # Two wire protocols on purpose: the production panel itself exercises
        # the provider seam (openai-compat AND native) across two model families.
        providers = PanelProviders(
            methods=OllamaOpenAIChat("llama3.1:8b"),
            novelty=OllamaNativeChat("qwen2:7b"),
            verifier=OllamaOpenAIChat("llama3.1:8b"),
            converger=OllamaNativeChat("qwen2:7b"),
        )
        review = run_panel(Path.cwd(), manuscript, providers, corpus=corpus)
        out = Path.cwd() / "artifacts" / corpus / f"review-{Path(names[0]).stem}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(review.model_dump_json(indent=1) + "\n")
        print(f"panel review of {review.manuscript_doi} (excluded: {review.excluded_docs})")
        for output in review.reviewer_outputs:
            print(f"  [{output.reviewer} · {output.model}] scores {output.scores} "
                  f"confidence {output.confidence} · {len(output.findings)} findings")
        supported = sum(1 for v in review.verdicts if v.verdict == "SUPPORTS")
        refuted = sum(1 for v in review.verdicts if v.verdict == "REFUTES")
        nei = sum(1 for v in review.verdicts if v.verdict == "NOT_ENOUGH_INFO")
        print(f"  claims: {supported} supported · {refuted} refuted · {nei} NEI "
              f"(swap-consistency {review.swap_consistency_rate} over n={review.n_swap_checked})")
        print(f"  deterministic lens: {len(review.deterministic_findings)} findings · "
              f"conflicts: {len(review.conflicts)} · tokens {review.total_tokens} · "
              f"wall {review.wall_s}s")
        print(f"  summary: {review.summary[:300]}")
        print(f"review: full record -> {out}")
        return 0
    if command not in PLANNED:
        print(f"peerpanel: unknown command {command!r}", file=sys.stderr)
        return 2
    print(
        f"peerpanel {command}: not built yet — it lands at its own checkpoint; "
        "see README for the build plan.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
