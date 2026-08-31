"""PeerPanel CLI. Commands land checkpoint by checkpoint; an unbuilt command says so honestly."""

from __future__ import annotations

import sys

BUILT: dict[str, str] = {}
PLANNED = ("quickstart", "demo", "corpus", "embeddings", "graph", "ablation", "eval")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] in ("-h", "--help"):
        print(f"peerpanel commands: {' '.join(PLANNED)}")
        return 0
    command = args[0]
    if command not in PLANNED:
        print(f"peerpanel: unknown command {command!r}", file=sys.stderr)
        return 2
    print(
        f"peerpanel {command}: not built yet — this is the C0 scaffold; "
        "see README for the build plan.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
