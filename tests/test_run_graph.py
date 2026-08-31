"""Per-run graph rebuild: exact exclusion, over the REAL committed CI artifacts.

The rebuild is what makes exclusion exact rather than approximate. These tests
pin both halves of that claim against real data: the entities the rebuild
removes are exactly the ones post-hoc filtering identifies, AND the rebuild
additionally removes the edge weight the excluded text contributed — which
filtering cannot reach.
"""

from __future__ import annotations

import time
from pathlib import Path

from peerpanel.embeddings import store
from peerpanel.embeddings.pipeline import ci_chunks
from peerpanel.graph.run_graph import build_run_graph
from peerpanel.retrieval import Index

ROOT = Path(__file__).resolve().parents[1]
TWIN = "PMC10729969"


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

    def test_rebuild_removes_twin_only_entities_and_their_edges(self) -> None:
        full, _fa, withheld_none = build_run_graph(ROOT, "ci", set())
        run, _ra, withheld = build_run_graph(ROOT, "ci", {TWIN})
        assert withheld_none == 0
        assert withheld > 0
        assert run.number_of_nodes() < full.number_of_nodes()
        assert full.number_of_nodes() - run.number_of_nodes() == 131
        assert full.number_of_edges() - run.number_of_edges() == 1247

    def test_rebuild_strips_twin_weight_from_surviving_edges(self) -> None:
        """The part post-hoc node filtering structurally cannot reach.

        These 38 edges join two entities that BOTH legitimately survive; the
        excluded text had inflated the weight of the link between them. Filtering
        nodes leaves that inflation in place — only a rebuild removes it.
        """
        full, _a, _w = build_run_graph(ROOT, "ci", set())
        run, _b, _w2 = build_run_graph(ROOT, "ci", {TWIN})
        lighter = [
            (a, b)
            for a, b in run.edges()
            if run.edges[a, b]["weight"] < full.edges[a, b]["weight"]
        ]
        assert len(lighter) == 38
        removed = sum(
            full.edges[a, b]["weight"] - run.edges[a, b]["weight"] for a, b in lighter
        )
        assert round(removed, 2) == 21.25

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
