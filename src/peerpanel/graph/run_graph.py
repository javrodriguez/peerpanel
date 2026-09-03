"""Per-run graph construction: the exact way to exclude a document.

`build_graph` is a pure aggregation over per-chunk extractions that are already
cached, so withholding a document costs a merge, not an extraction — measured
at 0.03 s for the CI corpus and 0.12 s more for Leiden. That makes the exact
thing affordable: rather than filtering entities out of a graph the excluded
text helped build, the graph is REBUILT for the run with that text's
extractions withheld. Nodes and — the part filtering cannot reach — the edge
weight the excluded text contributed both disappear.

What this does not rebuild is community summary TEXT, which is an LLM artifact
generated once over the full corpus. That is the one narrow residual, and
`graph-global` (which ranks over that text) is the one mode it touches.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from peerpanel.graph.build import build_graph
from peerpanel.graph.communities import detect
from peerpanel.graph.extract import EXTRACTION_PROVIDER_NAME, extract_many
from peerpanel.graph.pipeline import chunks_for
from peerpanel.providers.base import ChatResponse
from peerpanel.retrieval.base import doc_of

RESOLUTION = 1.0


class _CacheOnly:
    """Rebuilds must never call a model: every record is already committed.

    Extractions and community reports are filed under different cache identities
    (native wire and OpenAI wire respectively), so the stub is told which one it
    replays — a wrong name misses every record and raises, never silently re-keys.
    """

    def __init__(self, name: str = EXTRACTION_PROVIDER_NAME) -> None:
        self.name = name

    def chat(self, **_: object) -> ChatResponse:
        # The remedy names the road for the identity actually being replayed: sending a
        # reader to `make graph` because a community report is missing wastes an index
        # rebuild and does not produce the report.
        remedy = (
            "`make graph` / `make demo-index`"
            if self.name == EXTRACTION_PROVIDER_NAME
            else "`make summaries` / `make demo-summaries`"
        )
        what = "extraction" if self.name == EXTRACTION_PROVIDER_NAME else "community-report"
        raise RuntimeError(
            f"rebuilds replay committed records and never call a model; the {what} cache "
            f"({self.name}) is missing an entry — regenerate it with {remedy}."
        )


def build_run_graph(
    root: Path, corpus: str, exclude_docs: set[str]
) -> tuple[nx.Graph[str], dict[str, int], int]:
    """(graph, community assignment, chunks withheld) for one run.

    Deterministic and model-free: extractions come from the committed cache and
    Leiden is seeded.
    """
    chunks = chunks_for(root, corpus)
    cache_dir = root / "fixtures" / "extraction" / corpus
    extractions = extract_many(chunks, _CacheOnly(), cache_dir=cache_dir)
    kept = [e for e in extractions if doc_of(e.chunk_id) not in exclude_docs]
    withheld = len(extractions) - len(kept)
    graph = build_graph(kept)
    assignment = detect(graph, resolutions=(RESOLUTION,))[RESOLUTION]
    return graph, assignment, withheld
