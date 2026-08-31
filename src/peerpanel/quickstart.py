"""Tier-1 quickstart: the deterministic, no-LLM path a stranger runs first.

Everything here derives from COMMITTED bytes: the CI corpus, the embedding
fixture, and the recorded extraction/summary caches. The graph and community
reports are rebuilt through the exact production code path — the LLM provider
handed in is one that REFUSES to be called, so if any cache entry were
missing, the quickstart would fail loudly rather than quietly spend a model
call (or worse, pretend). Synthesis (live model answers) is tier 2: make demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from peerpanel.embeddings import store
from peerpanel.embeddings.pipeline import ci_chunks
from peerpanel.graph.build import build_graph, display_name, node_type, stats
from peerpanel.graph.communities import detect
from peerpanel.graph.extract import extract_many
from peerpanel.graph.summaries import summarise_communities
from peerpanel.providers.base import ChatResponse
from peerpanel.retrieval import BM25Retriever, GraphLocalRetriever, Index

QUERY = "How do yeast cells regulate sulfur and methionine metabolism?"


class _DeterministicOnly:
    """A ChatProvider that refuses: quickstart must run entirely from caches."""

    name = "ollama-openai:llama3.1:8b"  # the cache identity the fixtures were recorded under

    def chat(self, **_: Any) -> ChatResponse:
        raise RuntimeError(
            "quickstart is deterministic and never calls a model — a cache entry is "
            "missing; regenerate with `make graph` / `make summaries` (needs Ollama)."
        )


def run_quickstart(root: Path) -> int:
    chunks, findings = ci_chunks(root)
    n_docs = len({c.chunk_id.split(":", 1)[0] for c in chunks})
    print(f"corpus: {n_docs} docs · {len(chunks)} chunks · {len(findings)} sanitation findings")
    provider = _DeterministicOnly()
    extractions = extract_many(
        chunks, provider, cache_dir=root / "fixtures" / "extraction" / "ci"
    )
    graph = build_graph(extractions)
    graph_stats = stats(graph)
    print(
        f"graph: {graph_stats['nodes']} entities · {graph_stats['edges']} edges "
        "(rebuilt from recorded extractions)"
    )
    communities = detect(graph)
    assignment = communities[1.0]
    reports = summarise_communities(
        graph, assignment, 1.0, provider, cache_dir=root / "fixtures" / "summaries" / "ci"
    )
    print(
        f"communities: {len(set(assignment.values()))} at resolution 1.0 · "
        f"{len(reports)} reports (from recorded cache)"
    )
    chunk_ids, vectors = store.load(root / "fixtures" / "ci_embeddings.npz")
    if [c.chunk_id for c in chunks] != chunk_ids:
        raise RuntimeError("embedding fixture stale relative to the committed corpus")
    index = Index.build(chunks, vectors.astype("float32"))
    print(f'\nquery: "{QUERY}"')
    for name, retriever in (
        ("bm25", BM25Retriever(index)),
        ("graphrag-local (no embed blend in tier 1)", GraphLocalRetriever(index, graph)),
    ):
        hits = retriever.search(QUERY, k=3)
        print(f"  {name}:")
        for hit in hits:
            print(f"    #{hit.rank} {hit.chunk_id} (score {hit.score:.2f})")
    top = sorted(reports, key=lambda r: -r.size)[:3]
    print("\nlargest community reports (titles from the recorded cache):")
    for report in top:
        print(f"  [{report.community_id}] {report.title} ({report.size} members)")
    example = next(iter(sorted(graph.nodes())))
    print(f"\nexample entity: {display_name(graph, example)} ({node_type(graph, example)})")
    print("\ntier 1 complete — every number above derives from committed bytes; "
          "live synthesis is tier 2: `make demo` (needs Ollama).")
    return 0
