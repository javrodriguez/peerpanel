"""Graph assembly + Leiden tests — deterministic, no LLM, no network."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from peerpanel.graph.build import (
    build_graph,
    display_name,
    load_graph,
    node_type,
    save_graph,
    stats,
)
from peerpanel.graph.communities import detect, members
from peerpanel.graph.models import ChunkExtraction, Entity, Relation


def _ext(
    chunk_id: str, ents: list[tuple[str, str]], rels: list[tuple[str, str, str]]
) -> ChunkExtraction:
    return ChunkExtraction(
        chunk_id=chunk_id,
        entities=[Entity(name=n, type=t) for n, t in ents],
        relations=[Relation(source=s, predicate=p, target=t) for s, p, t in rels],
    )


class TestBuild:
    def test_case_variants_merge_and_vote(self) -> None:
        g = build_graph(
            [
                _ext(
                    "c1",
                    [("Met4", "gene_or_protein"), ("sulfur", "chemical")],
                    [("Met4", "activates", "sulfur")],
                ),
                _ext("c2", [("MET4", "gene_or_protein")], []),
            ]
        )
        assert g.has_node("met4")
        assert display_name(g, "met4") in ("Met4", "MET4")
        assert node_type(g, "met4") == "gene_or_protein"
        assert g.nodes["met4"]["chunk_ids"] == {"c1", "c2"}

    def test_relation_outweighs_co_mention(self) -> None:
        g = build_graph(
            [
                _ext("c1", [("A", "other"), ("B", "other"), ("C", "other")], [("A", "binds", "B")]),
            ]
        )
        assert g.edges["a", "b"]["weight"] == 1.25  # relation + co-mention
        assert g.edges["a", "c"]["weight"] == 0.25  # co-mention only
        assert g.edges["a", "b"]["predicates"]["binds"] == 1

    def test_self_relation_ignored(self) -> None:
        g = build_graph([_ext("c1", [("A", "other")], [("A", "is", "A")])])
        assert g.number_of_edges() == 0

    def test_roundtrip_persistence(self, tmp_path: Path) -> None:
        g = build_graph(
            [
                _ext(
                    "c1",
                    [("Met4", "gene_or_protein"), ("sulfur", "chemical")],
                    [("Met4", "activates", "sulfur")],
                ),
            ]
        )
        path = tmp_path / "graph.json"
        save_graph(g, path)
        g2 = load_graph(path)
        assert stats(g2) == stats(g)
        assert g2.nodes["met4"]["variants"] == g.nodes["met4"]["variants"]
        assert g2.edges["met4", "sulfur"]["predicates"] == g.edges["met4", "sulfur"]["predicates"]


class TestCommunities:
    def _two_cluster_graph(self) -> nx.Graph[str]:
        exts = []
        for i, cluster in enumerate((["A1", "A2", "A3", "A4"], ["B1", "B2", "B3", "B4"])):
            for j in range(len(cluster)):
                for k in range(j + 1, len(cluster)):
                    exts.append(
                        _ext(
                            f"c{i}{j}{k}",
                            [(cluster[j], "other"), (cluster[k], "other")],
                            [(cluster[j], "links", cluster[k])],
                        )
                    )
        exts.append(_ext("bridge", [("A1", "other"), ("B1", "other")], []))
        return build_graph(exts)

    def test_two_clusters_found_and_deterministic(self) -> None:
        g = self._two_cluster_graph()
        first = detect(g, resolutions=(1.0,))[1.0]
        second = detect(g, resolutions=(1.0,))[1.0]
        assert first == second  # seeded
        groups = members(first)
        assert len(groups) == 2
        sides = sorted(frozenset(m) for m in groups.values())
        assert {"a1", "a2", "a3", "a4"} in [set(s) for s in sides]

    def test_finer_resolution_never_coarser(self) -> None:
        g = self._two_cluster_graph()
        result = detect(g, resolutions=(0.3, 3.0))
        assert len(members(result[3.0])) >= len(members(result[0.3]))

    def test_empty_graph(self) -> None:
        assert detect(nx.Graph(), resolutions=(1.0,)) == {1.0: {}}
