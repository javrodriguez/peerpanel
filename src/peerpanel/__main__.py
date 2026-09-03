"""PeerPanel CLI. Commands land checkpoint by checkpoint; an unbuilt command says so honestly."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from peerpanel.graph.pipeline import DemoCorpusMissing

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


def _run_suffix(args: list[str]) -> str:
    """`--run N` names a repeat of a published record: run 1 has no suffix, run N>1
    is `-runN`, so a second independent run sits beside the first instead of
    overwriting it."""
    if "--run" not in args:
        return ""
    run = int(args[args.index("--run") + 1])
    return "" if run == 1 else f"-run{run}"


_VALUE_OPTIONS = ("--corpus", "--run")  # options that consume the next argument


def _positionals(args: list[str]) -> list[str]:
    out: list[str] = []
    skip = False
    for arg in args:
        if skip:
            skip = False
        elif arg in _VALUE_OPTIONS:
            skip = True
        elif not arg.startswith("--"):
            out.append(arg)
    return out


def _where(args: list[str], out: Path) -> bool:
    """`--where`: print the file this invocation would write, relative to the working
    directory, and do no work. It is the model-free half of every regenerate command
    the prose cites: a test runs the cited command with --where and proves the path it
    names is the committed file, so a row can never point at a command that writes
    somewhere else (the drift class this repository has already shipped three times).
    """
    if "--where" not in args:
        return False
    print(out.relative_to(Path.cwd()).as_posix())
    return True


def _manuscript_arg(args: list[str], default: str, command: str) -> Path | None:
    names = _positionals(args)
    stem = Path(names[0]).stem if names else default
    manuscript = Path.cwd() / "manuscripts" / f"{stem}.txt"
    if not manuscript.exists():
        print(f"{command}: no manuscript at {manuscript}", file=sys.stderr)
        return None
    return manuscript


def _graph(args: list[str]) -> int:
    root = Path.cwd()
    sub = args[0] if args else "build"
    corpus = _corpus_arg(args)
    if sub == "build":
        from peerpanel.graph.extract import EXTRACTION_MODEL
        from peerpanel.graph.pipeline import graph_build, write_demo_embedding_fixture
        from peerpanel.providers import OllamaNativeChat, OllamaNativeEmbed

        # --publish also writes the tracked build record the prose cites; without it
        # the stats stay under artifacts/ (ignored) with the graph they describe.
        publish = root / "results" / f"build-stats-{corpus}.json" if "--publish" in args else None
        if _where(args, publish or root / "artifacts" / corpus / "build_stats.json"):
            return 0
        if corpus == "demo":
            digest = write_demo_embedding_fixture(root, OllamaNativeEmbed())
            print(f"demo embedding fixture written, sha256 {digest}")
        # The native wire sizes the window per chunk (providers/context.py, D19);
        # the community reports below stay on the OpenAI wire, whose prompts fit
        # Ollama's default window by construction.
        result = graph_build(root, OllamaNativeChat(EXTRACTION_MODEL), corpus=corpus)
        print(json.dumps(result, indent=1, sort_keys=True))
        if publish is not None:
            publish.parent.mkdir(parents=True, exist_ok=True)
            publish.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
            print(f"graph build: published -> {publish}")
        truncated = result.get("truncated_chunks")
        if isinstance(truncated, int) and truncated:
            print(f"graph build: {truncated} chunk(s) recorded as truncated", file=sys.stderr)
        return 0
    if sub == "summaries":
        # --where before anything else: the door must do no work, and the cache README
        # written below is work. Without this door the regenerate test, which runs every
        # cited command with --where, ran the real community-report pass — 122 live model
        # calls inside `make test`, and a hard failure anywhere without Ollama.
        summaries_out = root / "artifacts" / corpus / "summaries.json"
        if _where(args, summaries_out):
            return 0

        from peerpanel.graph.build import load_graph
        from peerpanel.graph.summaries import (
            SUMMARY_MODEL,
            summarise_communities,
            write_cache_readme,
        )
        from peerpanel.providers import OllamaOpenAIChat

        graph = load_graph(root / "artifacts" / corpus / "graph.json")
        communities = json.loads((root / "artifacts" / corpus / "communities.json").read_text())
        provider = OllamaOpenAIChat(SUMMARY_MODEL)  # cache identity: SUMMARY_PROVIDER_NAME
        cache = root / "fixtures" / "summaries" / corpus
        write_cache_readme(
            cache, corpus, "make summaries" if corpus == "ci" else "make demo-summaries"
        )
        # --resolution limits which Leiden levels get LLM reports. The hierarchy is
        # always detected at every level (communities.json); reports cost one model
        # call each, so by default only the level retrieval actually reads is
        # summarised. Stated in the README rather than implied.
        wanted = (
            {args[args.index("--resolution") + 1]} if "--resolution" in args else set(communities)
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
        out = summaries_out
        out.write_text(json.dumps(all_reports, indent=1, sort_keys=True) + "\n")
        truncated = sum(1 for r in all_reports if bool(r.get("truncated")))
        if truncated:
            print(f"graph summaries: {truncated} report(s) truncated", file=sys.stderr)
        print(f"graph summaries: {len(all_reports)} reports -> {out}")
        return 0
    print(f"graph: unknown subcommand {sub!r} (build | summaries)", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    try:
        return _main(sys.argv[1:] if argv is None else argv)
    except DemoCorpusMissing as missing:
        # A documented command meeting a fetchable prerequisite should say what to
        # run, not print a traceback at someone following the README.
        print(f"\n{missing}", file=sys.stderr)
        return 3


def _main(args: list[str]) -> int:
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
        from peerpanel.embeddings.query_cache import CachedQueryEmbedder
        from peerpanel.evals.ablation import render_table, run_ablation

        corpus = _corpus_arg(rest)
        # Default output is the untracked working path, so running the documented
        # command never dirties committed evidence (latencies are machine-dependent
        # and can never be byte-reproducible). --publish updates results/, which is
        # tracked because a measurement nobody can see is not published.
        if "--publish" in rest:
            out = Path.cwd() / "results" / f"ablation-{corpus}.json"
        else:
            out = Path.cwd() / "artifacts" / corpus / "ablation.json"
        if _where(rest, out):
            return 0
        # Committed query vectors by default, so this runs from a clean clone with
        # no model; --live regenerates them against the real embedder.
        live = None
        if "--live" in rest:
            from peerpanel.providers import OllamaNativeEmbed

            live = OllamaNativeEmbed()
        embedder = CachedQueryEmbedder(Path.cwd(), live=live)
        report = run_ablation(Path.cwd(), embedder, corpus=corpus)
        if live is not None:
            digest = embedder.save(live.name)
            print(f"query-embedding fixture written, sha256 {digest}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report.model_dump_json(indent=1) + "\n")
        table = render_table(report)
        print(table)
        print(f"ablation: full report -> {out}")
        return 0
    if command == "demo":
        from peerpanel.demo import MANUSCRIPT, replay, run_demo
        from peerpanel.orchestration import PanelProviders

        # The same door `graph summaries` needed, for the same reason: the regenerate test
        # runs every command a results-table row cites with --where. Without it, the day a
        # demo capture is committed that test starts a full live panel run inside
        # `make test` — and inside CI, which has no models at all.
        # run_demo's own default is the demo corpus, not the "ci" default _corpus_arg
        # carries for every other command — a door that names the wrong path is worse
        # than no door, because a test would then prove the wrong file.
        corpus = rest[rest.index("--corpus") + 1] if "--corpus" in rest else "demo"
        if _where(rest, Path.cwd() / "artifacts" / corpus / f"demo-review-{MANUSCRIPT}.json"):
            return 0
        if "--replay" in rest:
            return replay(Path.cwd())
        return run_demo(Path.cwd(), PanelProviders.local_default())
    if command == "eval":
        from peerpanel.evals.planted_eval import render_table as render_planted
        from peerpanel.evals.planted_eval import run_planted_eval
        from peerpanel.orchestration import PanelProviders
        from peerpanel.providers import OllamaNativeChat, OllamaNativeEmbed

        manuscript = _manuscript_arg(rest, "caprin-heterochromatin", "eval")
        if manuscript is None:
            return 2
        corpus = _corpus_arg(rest) if "--corpus" in rest else "demo"
        # --publish writes the tracked record the prose cites, named by subject and
        # run; without it the report stays under artifacts/ (ignored).
        if "--publish" in rest:
            out = Path.cwd() / "results" / f"planted-eval-{manuscript.stem}{_run_suffix(rest)}.json"
        else:
            out = Path.cwd() / "artifacts" / corpus / f"planted-eval-{manuscript.stem}.json"
        if _where(rest, out):
            return 0
        planted_report = run_planted_eval(
            Path.cwd(),
            manuscript,
            PanelProviders.local_default(),
            # The same model and wire as the panel's methods reviewer, so the
            # arms differ in architecture, never in transport or window.
            baseline_provider=OllamaNativeChat("llama3.1:8b"),
            embedder=OllamaNativeEmbed(),
            corpus=corpus,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(planted_report.model_dump_json(indent=1) + "\n")
        print(render_planted(planted_report))
        print(f"eval: full report -> {out}")
        return 0
    if command == "review":
        from peerpanel.orchestration import PanelProviders, run_panel

        manuscript = _manuscript_arg(rest, "met17-auxotroph", "review")
        if manuscript is None:
            return 2
        corpus = _corpus_arg(rest) if "--corpus" in rest else "demo"
        if "--publish" in rest:
            out = Path.cwd() / "results" / f"panel-review-{manuscript.stem}{_run_suffix(rest)}.json"
        else:
            out = Path.cwd() / "artifacts" / corpus / f"review-{manuscript.stem}.json"
        if _where(rest, out):
            return 0
        review = run_panel(Path.cwd(), manuscript, PanelProviders.local_default(), corpus=corpus)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(review.model_dump_json(indent=1) + "\n")
        print(f"panel review of {review.manuscript_doi} (excluded: {review.excluded_docs})")
        for output in review.reviewer_outputs:
            print(
                f"  [{output.reviewer} · {output.model}] scores {output.scores} "
                f"confidence {output.confidence} · {len(output.findings)} findings"
            )
        supported = sum(1 for v in review.verdicts if v.verdict == "SUPPORTS")
        refuted = sum(1 for v in review.verdicts if v.verdict == "REFUTES")
        nei = sum(1 for v in review.verdicts if v.verdict == "NOT_ENOUGH_INFO")
        print(
            f"  claims: {supported} supported · {refuted} refuted · {nei} NEI "
            f"(swap-consistency {review.swap_consistency_rate} over n={review.n_swap_checked})"
        )
        print(
            f"  deterministic lens: {len(review.deterministic_findings)} findings · "
            f"conflicts: {len(review.conflicts)} · tokens {review.total_tokens} · "
            f"wall {review.wall_s}s"
        )
        calls = review.model_calls
        print(
            f"  model calls: {calls.calls} · largest prompt {calls.largest_prompt_tokens} "
            f"tokens · smallest window {calls.smallest_context} (every prompt read whole)"
        )
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
