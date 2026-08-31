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


def _graph(args: list[str]) -> int:
    root = Path.cwd()
    sub = args[0] if args else "build"
    if sub == "build":
        from peerpanel.graph.pipeline import ci_graph_build
        from peerpanel.providers import OllamaOpenAIChat

        result = ci_graph_build(root, OllamaOpenAIChat("llama3.1:8b"))
        print(json.dumps(result, indent=1, sort_keys=True))
        truncated = result.get("truncated_chunks")
        if isinstance(truncated, int) and truncated:
            print(f"graph build: {truncated} chunk(s) recorded as truncated", file=sys.stderr)
        return 0
    if sub == "summaries":
        print(
            "peerpanel graph summaries: not built yet — it lands at checkpoint C2/T2.3.",
            file=sys.stderr,
        )
        return 3
    print(f"graph: unknown subcommand {sub!r} (build | summaries)", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print(f"peerpanel commands: {' '.join(PLANNED)}")
        return 0
    command, rest = args[0], args[1:]
    if command == "corpus":
        return _corpus(rest)
    if command == "embeddings":
        return _embeddings(rest)
    if command == "graph":
        return _graph(rest)
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
