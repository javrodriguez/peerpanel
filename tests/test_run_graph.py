"""Per-run graph rebuild: exact exclusion, over the REAL committed CI artifacts.

The rebuild is what makes exclusion exact rather than approximate. These tests pin both
halves of that claim against real data: the entities the rebuild removes are exactly the
ones post-hoc filtering identifies, AND the rebuild additionally removes the edge weight
the excluded text contributed — which filtering cannot reach.

Every expected value here is DERIVED from an independent mechanism — the full graph's own
chunk provenance, or a graph built from the twin's extractions alone — never from the
rebuild being tested and never pasted from a run's output. A constant copied from the
thing it checks is D14's defect wearing a test's clothes, and this file carried four such
constants until 2026-09-02.

Both mutations were run against these bytes, and each is red where it should be:

  * build the full graph and REMOVE the twin's nodes, instead of rebuilding from the kept
    extractions -> 4 red (both edge tests, the weight identity, and the excluded-evidence
    check) while the node test stays GREEN. That is the point: filtering gets the entity
    set exactly right, and the edges wrong.
  * make the exclusion a no-op (`kept = extractions`) -> 7 of 8 red.

If a change here leaves both mutations green, the assertions have stopped measuring
anything, whatever their numbers say.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import networkx as nx
import pytest

from peerpanel.embeddings import store
from peerpanel.embeddings.pipeline import ci_chunks
from peerpanel.graph import build as build_module
from peerpanel.graph.build import build_graph
from peerpanel.graph.extract import ChunkExtraction, extract_many
from peerpanel.graph.pipeline import chunks_for
from peerpanel.graph.run_graph import _CacheOnly, build_run_graph, doc_of
from peerpanel.retrieval import Index

ROOT = Path(__file__).resolve().parents[1]
TWIN = "PMC10729969"


def _extractions() -> list[ChunkExtraction]:
    """The committed CI extractions, replayed — no model, no network."""
    return extract_many(
        chunks_for(ROOT, "ci"), _CacheOnly(), cache_dir=ROOT / "fixtures" / "extraction" / "ci"
    )


def _twin_graph() -> nx.Graph[str]:
    """A graph built from the twin's extractions ALONE. Independent of the exclusion
    path: it is how much of the full graph the twin was responsible for, computed the
    long way round."""
    return build_graph([e for e in _extractions() if doc_of(e.chunk_id) == TWIN])


def _twin_evidence_only(full: nx.Graph[str]) -> set[str]:
    """Entities the full graph saw ONLY in the twin's chunks — derived from `full`, not
    from the rebuild whose behaviour is under test."""
    return {
        n
        for n, at in full.nodes(data=True)
        if all(c.startswith(f"{TWIN}:") for c in at["chunk_ids"])
    }


class TestRunGraphRebuild:
    def test_rebuild_is_model_free_and_fast(self) -> None:
        t0 = time.monotonic()
        graph, assignment, withheld = build_run_graph(ROOT, "ci", {TWIN})
        elapsed = time.monotonic() - t0
        assert withheld > 0
        assert graph.number_of_nodes() > 0
        assert assignment
        # A pure merge over cached extractions plus seeded Leiden, measured at ~0.2s.
        # The bound is loose enough never to flake on a slow runner and tight enough
        # to fail loudly if a model call is ever reintroduced into this path.
        assert elapsed < 5

    def test_rebuild_removes_exactly_the_entities_whose_evidence_was_withheld(self) -> None:
        """Two independent mechanisms, one answer.

        The expected removal is derived from the FULL graph alone — an entity goes if
        every chunk it was seen in belongs to the twin — and compared with what the
        rebuild actually did. Deriving it from `full - run` instead would be a tautology
        that no mutation can turn red (D11, and round 1's F8/F9/F11).

        Measured at this commit: 142 of 1,759 entities.
        """
        full, _fa, withheld_none = build_run_graph(ROOT, "ci", set())
        run, _ra, withheld = build_run_graph(ROOT, "ci", {TWIN})
        assert withheld_none == 0
        assert withheld > 0
        twin_only = _twin_evidence_only(full)
        assert twin_only, "the twin contributed no entity of its own — the check is vacuous"
        assert set(full) - set(run) == twin_only

    def test_rebuild_removes_exactly_the_edges_the_withheld_text_carried(self) -> None:
        """An edge goes for one of two reasons, and both are derived independently: an
        endpoint was removed, or every unit of its weight came from the twin's chunks.
        The second set is computed from a graph built from the twin's extractions alone,
        which never passes through the exclusion path being tested.

        Measured at this commit: 1,352 edges gone, 1,260 by endpoint and 92 that joined
        two surviving entities and existed only because the twin mentioned them together.
        """
        full, _a, _w = build_run_graph(ROOT, "ci", set())
        run, _b, _w2 = build_run_graph(ROOT, "ci", {TWIN})
        twin_graph = _twin_graph()
        removed_nodes = _twin_evidence_only(full)
        gone = {frozenset(e) for e in full.edges()} - {frozenset(e) for e in run.edges()}
        by_endpoint = {
            frozenset((a, b)) for a, b in full.edges() if a in removed_nodes or b in removed_nodes
        }
        all_weight_from_twin = {
            frozenset((a, b))
            for a, b in full.edges()
            if a not in removed_nodes
            and b not in removed_nodes
            and twin_graph.has_edge(a, b)
            and full.edges[a, b]["weight"] == pytest.approx(twin_graph.edges[a, b]["weight"])
        }
        assert all_weight_from_twin, "no edge existed only because of the twin"
        assert gone == by_endpoint | all_weight_from_twin

    def test_a_surviving_edge_loses_exactly_the_weight_the_twin_contributed(self) -> None:
        """The part post-hoc node filtering structurally cannot reach.

        These edges join two entities that BOTH legitimately survive; the excluded text
        had inflated the weight of the link between them. The expected loss is not a
        constant — it is the weight of that same edge in a graph built from the twin's
        extractions alone, which is an independent computation of the same quantity.

        Measured at this commit: 48 edges, 27.25 weight units, and the identity holds for
        every one of the 48.
        """
        full, _a, _w = build_run_graph(ROOT, "ci", set())
        run, _b, _w2 = build_run_graph(ROOT, "ci", {TWIN})
        twin_graph = _twin_graph()
        lighter = [
            (a, b) for a, b in run.edges() if run.edges[a, b]["weight"] < full.edges[a, b]["weight"]
        ]
        assert lighter, (
            "no surviving edge lost weight: either the rebuild stopped stripping it, or "
            "this corpus no longer has a twin sharing edges with surviving entities — "
            "both are findings, neither is a number to update"
        )
        for a, b in lighter:
            lost = full.edges[a, b]["weight"] - run.edges[a, b]["weight"]
            assert twin_graph.has_edge(a, b), (a, b)
            assert lost == pytest.approx(twin_graph.edges[a, b]["weight"]), (a, b)

    def test_post_hoc_node_filtering_cannot_produce_this_graph(self) -> None:
        """The mutation, kept as a test rather than a claim in a docstring.

        Filtering the full graph's nodes — the obvious cheap alternative to rebuilding —
        gets the entity set exactly right and the EDGES wrong, both by leaving edges that
        existed only through the twin and by leaving twin weight on edges that survive.
        If this test ever passes trivially, the rebuild has degraded into filtering and
        the exclusion claim has lost its teeth.
        """
        full, _a, _w = build_run_graph(ROOT, "ci", set())
        run, _b, _w2 = build_run_graph(ROOT, "ci", {TWIN})
        filtered = build_graph(_extractions())
        filtered.remove_nodes_from(_twin_evidence_only(full))
        assert set(filtered) == set(run)  # nodes: filtering is enough
        assert filtered.number_of_edges() > run.number_of_edges()  # edges: it is not
        heavier = [
            (a, b)
            for a, b in filtered.edges()
            if run.has_edge(a, b) and filtered.edges[a, b]["weight"] > run.edges[a, b]["weight"]
        ]
        assert heavier, "filtering left no inflated edge — then the rebuild buys nothing"

    def test_rebuild_agrees_with_live_nodes_on_which_entities_survive(self) -> None:
        """Two independent mechanisms, one answer — a contract-drift check."""
        full, _a, _w = build_run_graph(ROOT, "ci", set())
        run, _b, _w2 = build_run_graph(ROOT, "ci", {TWIN})
        chunks, _ = ci_chunks(ROOT)
        _ids, vectors = store.load(ROOT / "fixtures" / "ci_embeddings.npz")
        index = Index.build(chunks, vectors.astype("float32"), exclude_docs={TWIN})
        assert index.live_nodes(full) == set(run.nodes())

    def test_no_surviving_entity_carries_excluded_evidence(self) -> None:
        run, _a, _w = build_run_graph(ROOT, "ci", {TWIN})
        for _node, attrs in run.nodes(data=True):
            assert all(not cid.startswith(f"{TWIN}:") for cid in attrs["chunk_ids"])

    def test_deterministic_across_calls(self) -> None:
        a, aa, _ = build_run_graph(ROOT, "ci", {TWIN})
        b, bb, _ = build_run_graph(ROOT, "ci", {TWIN})
        assert set(a.nodes()) == set(b.nodes())
        assert a.number_of_edges() == b.number_of_edges()
        assert aa == bb  # seeded Leiden


def _demo_graph() -> nx.Graph[str]:
    """The demo graph, from the COMMITTED extraction fixtures only.

    `build_run_graph` cannot be used here: it re-chunks `corpus/demo/`, which is
    gitignored and fetched, so the recipe LIMITATIONS.md used to give raised
    `DemoCorpusMissing` from a clean clone (round-3 report-2 F5). Merging the committed
    per-chunk records is a pure aggregation and reproduces `build-stats-demo.json`
    exactly — that is the offline road, and it is the one the bindings take.
    """
    records = [
        ChunkExtraction.model_validate_json(path.read_text())
        for path in sorted((ROOT / "fixtures" / "extraction" / "demo").glob("*.json"))
    ]
    assert records, "no committed demo extractions — this binding would check nothing"
    return build_graph(records)


class TestTheBuildRecordsAreThisCodesOutput:
    """The published index numbers, recomputed from the committed extractions.

    `results/build-stats-*.json` is quoted in RESULTS.md's index table, in the README
    and in LIMITATIONS.md. Nothing compared it with what this code produces from the
    fixtures beside it, so a rebuild that changed the graph would have left three
    documents describing the old one.
    """

    def test_ci_graph_stats_match_the_build_record(self) -> None:
        record = json.loads((ROOT / "results" / "build-stats-ci.json").read_text())
        graph, assignment, withheld = build_run_graph(ROOT, "ci", set())
        assert withheld == 0
        assert graph.number_of_nodes() == record["nodes"], (
            f"the CI extractions merge to {graph.number_of_nodes()} entities; "
            f"build-stats-ci.json says {record['nodes']}"
        )
        assert graph.number_of_edges() == record["edges"], (
            f"the CI extractions merge to {graph.number_of_edges()} edges; "
            f"build-stats-ci.json says {record['edges']}"
        )
        communities = len(set(assignment.values()))
        published = record["communities_per_resolution"]["1.0"]
        assert communities == published, (
            f"seeded Leiden finds {communities} communities at resolution 1.0; "
            f"build-stats-ci.json says {published}"
        )

    def test_relation_share_docstring_is_bound(self) -> None:
        """`graph/build.py`'s own docstring says how much of the graph is bare
        adjacency — the sentence LIMITATIONS.md and the README both lean on. It was
        written against a pre-D19 index and drifted by four thousand edges with nothing
        to notice, because a docstring is prose that no test reads. This one does.
        """
        graph = _demo_graph()
        total = graph.number_of_edges()
        co_only = sum(1 for _a, _b, at in graph.edges(data=True) if not at["predicates"])
        with_relation = total - co_only
        share = 100 * co_only / total
        measured = (
            f"measured: {share:.1f}% of edges ({co_only:,} of {total:,}) are co-mention "
            f"only, {with_relation:,} carry an extracted relation"
        )
        doc = build_module.__doc__ or ""

        percent = re.search(r"(\d+(?:\.\d+)?)\s*%\s+of edges", doc)
        assert percent, f"build.py's docstring states no co-mention share — {measured}"
        decimals = len(percent.group(1).split(".")[1]) if "." in percent.group(1) else 0
        assert float(percent.group(1)) == pytest.approx(share, abs=0.5 * 10.0**-decimals + 1e-9), (
            f"build.py says {percent.group(1)}% of edges are co-mention only; {measured}"
        )

        pair = re.search(r"\(([\d,]+) of ([\d,]+)\)", doc)
        assert pair, f"build.py's docstring states no edge counts — {measured}"
        assert [int(g.replace(",", "")) for g in pair.groups()] == [co_only, total], (
            f"build.py says {pair.group(0)}; {measured}"
        )

        relations = re.search(r"([\d,]+) carry an extracted relation", doc)
        assert relations, f"build.py's docstring states no relation count — {measured}"
        assert int(relations.group(1).replace(",", "")) == with_relation, (
            f"build.py says {relations.group(1)} edges carry a relation; {measured}"
        )

    def test_the_demo_merge_reproduces_the_published_index(self) -> None:
        """The offline road LIMITATIONS.md now names, proven to arrive where it claims:
        merging the committed demo extractions gives the published graph exactly."""
        record = json.loads((ROOT / "results" / "build-stats-demo.json").read_text())
        graph = _demo_graph()
        assert (graph.number_of_nodes(), graph.number_of_edges()) == (
            record["nodes"],
            record["edges"],
        ), (
            f"the committed demo extractions merge to {graph.number_of_nodes()} entities "
            f"and {graph.number_of_edges()} edges; build-stats-demo.json says "
            f"{record['nodes']} and {record['edges']}"
        )
