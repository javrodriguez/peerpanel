"""PeerPanel CLI. Commands land checkpoint by checkpoint; an unbuilt command says so honestly."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from peerpanel.graph.pipeline import DemoCorpusMissing

PLANNED = (
    "quickstart",
    "demo",
    "corpus",
    "embeddings",
    "graph",
    "ablation",
    "eval",
    "results",
)


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
        # --publish also writes the tracked run record the prose cites: which wire and
        # model produced the reports, how many calls THIS run made, and the per-call
        # margin between prompt and window. Until round 3 this layer — the only input to
        # graphrag-global's ranking, on the one wire that cannot set its window — shipped
        # no record of its run conditions at all.
        stats_out = root / "results" / f"summaries-stats-{corpus}.json"
        if "--where" in args:
            for path in (summaries_out, *([stats_out] if "--publish" in args else [])):
                print(path.relative_to(Path.cwd()).as_posix())
            return 0

        from peerpanel.agents.schemas import CallStats
        from peerpanel.graph.build import load_graph
        from peerpanel.graph.summaries import (
            SUMMARY_MODEL,
            SummaryRunStats,
            summarise_communities,
            write_cache_readme,
        )
        from peerpanel.providers import OllamaOpenAIChat
        from peerpanel.providers.base import TokenLedger

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
        ledger = TokenLedger()
        run_stats = SummaryRunStats()
        for resolution, assignment in communities.items():
            if resolution not in wanted:
                continue
            typed = {n: int(c) for n, c in assignment.items()}
            reports = summarise_communities(
                graph,
                typed,
                float(resolution),
                provider,
                cache_dir=cache,
                ledger=ledger,
                stats=run_stats,
            )
            all_reports.extend(r.model_dump() for r in reports)
            print(f"resolution {resolution}: {len(reports)} community reports")
        out = summaries_out
        out.write_text(json.dumps(all_reports, indent=1, sort_keys=True) + "\n")
        truncated = sum(1 for r in all_reports if bool(r.get("truncated")))
        if truncated:
            print(f"graph summaries: {truncated} report(s) truncated", file=sys.stderr)
        print(f"graph summaries: {len(all_reports)} reports -> {out}")
        if "--publish" in args:
            record = {
                "corpus": corpus,
                "provider": provider.name,
                "resolutions": sorted(wanted, key=float),
                "reports": len(all_reports),
                "reports_truncated": truncated,
                # How the reports were produced, so a reader can tell a cold run from a
                # warm one without leaving the record. These two always sum to `reports`.
                # `from_cache` is not only the committed cache: the cache is keyed by a
                # community's content and not by its resolution, so the same community
                # recurs across Leiden levels and a run that began with an empty cache
                # still reads back entries it wrote itself minutes earlier.
                "reports_generated": run_stats.generated,
                "reports_from_cache": run_stats.from_cache,
                # calls THIS run made; a report read from cache makes none
                "model_calls": CallStats.from_ledger(ledger).model_dump(),
            }
            stats_out.parent.mkdir(parents=True, exist_ok=True)
            stats_out.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
            print(f"graph summaries: published -> {stats_out}")
        return 0
    print(f"graph: unknown subcommand {sub!r} (build | summaries)", file=sys.stderr)
    return 2


# The rule text `results/derived-fields.json` opens with. Written here only when the
# manifest does not exist yet; the committed file is the authority once it does.
_DERIVED_FIELDS_RULE = (
    "every entry names a field written by derivation, not measurement; "
    "tests/test_run_conditions.py proves each equals the emitter's own derivation"
)

# The regenerate command for each record this door may derive a field onto. A CLOSED
# map on purpose: an entry in derived-fields.json is read as the instruction that
# reproduces the record's shape, and a manifest carrying an invented `make` line is the
# documented-command-that-does-not-reproduce defect this repository has shipped before.
_DERIVE_REGENERATE = {
    "build-stats-ci.json": "make graph after emptying fixtures/extraction/ci",
    "build-stats-demo.json": "make demo-index",
}


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _results(args: list[str]) -> int:
    """`results derive-window-sources <record>...` — the disclosed derivation door.

    The two index-build records are NOT re-run to gain `model_calls.window_sources`: a
    non-deterministic re-extraction would move every downstream number in order to write
    a string the code path already determines from the wire the record names. So the
    field is written by the emitter's own derivation (`evals.records.derive_window_sources`
    — the same `provider` prefix -> constant mapping `graph build --publish` emits), and
    NOTHING else is written into the record: a provenance key inside a record the emitter
    cannot produce is a mock by drift. That the field was derived rather than measured is
    disclosed BESIDE the record, in `results/derived-fields.json`, which is in the
    evaluators' scope. Idempotent: a second run writes nothing and keeps the first run's
    date, because when the field was derived is a fact about the record, not about the
    last time someone re-ran the door.
    """
    root = Path.cwd()
    sub = args[0] if args else ""
    if sub != "derive-window-sources":
        print(f"results: unknown subcommand {sub!r} (derive-window-sources)", file=sys.stderr)
        return 2
    targets = [root / name for name in _positionals(args[1:])]
    manifest_path = root / "results" / "derived-fields.json"
    if not targets:
        print(
            "results derive-window-sources: name at least one record, e.g. "
            "results/build-stats-ci.json",
            file=sys.stderr,
        )
        return 2
    if "--where" in args:
        # Every path this invocation would write: the records themselves and the
        # manifest that discloses the derivation. --where does no work, so it cannot
        # know which of them a run would actually change.
        for path in (*targets, manifest_path):
            print(_rel(path, root))
        return 0

    from datetime import UTC, datetime

    from peerpanel.evals.records import WINDOW_SOURCES_FIELD, derive_window_sources

    missing = [_rel(p, root) for p in targets if not p.exists()]
    if missing:
        print(f"results derive-window-sources: no such record(s): {missing}", file=sys.stderr)
        return 2
    unmapped = [p.name for p in targets if p.name not in _DERIVE_REGENERATE]
    if unmapped:
        print(
            f"results derive-window-sources: no regenerate command is recorded for "
            f"{unmapped} — add it to _DERIVE_REGENERATE beside this door rather than "
            "letting the manifest carry a guess",
            file=sys.stderr,
        )
        return 2

    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {"_rule": _DERIVED_FIELDS_RULE, "records": {}}
    )
    records: dict[str, Any] = manifest.setdefault("records", {})
    today = datetime.now(UTC).date().isoformat()
    written: list[str] = []
    for path in targets:
        before = path.read_text()
        derived = derive_window_sources(json.loads(before))
        after = json.dumps(derived, indent=1, sort_keys=True) + "\n"
        if after != before:
            path.write_text(after)
            written.append(_rel(path, root))
        previous: dict[str, Any] = records.get(path.name) or {}
        records[path.name] = {
            # Imported, never retyped: the manifest's `field` and the field the
            # derivation writes are one name, and a test reads both.
            "field": WINDOW_SOURCES_FIELD,
            "derived_from": "provider",
            "derived_on": previous.get("derived_on", today),
            "re_measured": False,
            "regenerate": _DERIVE_REGENERATE[path.name],
        }
    manifest_text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    if not manifest_path.exists() or manifest_text != manifest_path.read_text():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(manifest_text)
        written.append(_rel(manifest_path, root))
    print(
        f"results derive-window-sources: {len(targets)} record(s) read · "
        + (f"wrote {', '.join(written)}" if written else "nothing changed (already derived)")
    )
    return 0


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
    if command == "results":
        return _results(rest)
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
        if "--collisions" in rest:
            # The model-free half of this evaluation's soundness: for every planted
            # token, which corpus documents already contain it. Committed so the
            # collision guard the headline leans on runs in EVERY clean clone rather
            # than skipping wherever the fetched corpus is absent (round-3 F12).
            collisions_out = Path.cwd() / "corpus" / "token_collisions.json"
            if _where(rest, collisions_out):
                return 0
            from peerpanel.evals.collisions import write_token_collisions

            collisions_corpus = _corpus_arg(rest) if "--corpus" in rest else "demo"
            subjects = sorted(p.stem for p in (Path.cwd() / "manuscripts").glob("*.txt"))
            out_path = write_token_collisions(Path.cwd(), collisions_corpus, subjects)
            print(f"eval collisions: {len(subjects)} manuscript(s) -> {out_path}")
            return 0

        from peerpanel.evals.planted_eval import BASELINE_MODELS, run_planted_eval
        from peerpanel.evals.planted_eval import render_table as render_planted
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
            # D-2: ONE single-agent arm per named local model, published as its own
            # row, so "pass rates for both local models" is a rate per model rather
            # than one number over a mixture. Both ride the native wire — the same
            # transport and per-call window sizing as the panel — so the arms differ
            # in architecture and model, never in transport or window.
            baseline_providers=[OllamaNativeChat(model) for model in BASELINE_MODELS],
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
        # The margin is the proof; the largest prompt and the smallest window are usually
        # different calls, and printing only those two beside "read whole" made the log
        # read as a cut prompt with a parenthesis denying it (round-3 review).
        margin = calls.smallest_headroom_tokens
        verdict = (
            f"smallest margin {margin} tokens — every prompt read whole"
            if margin is not None and margin > 0
            else f"smallest margin {margin} — NOT every prompt read whole"
        )
        print(
            f"  model calls: {calls.calls} · largest prompt {calls.largest_prompt_tokens} "
            f"tokens · smallest window {calls.smallest_context} · {verdict}"
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
