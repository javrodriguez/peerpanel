"""Graph assembly: merge per-chunk extractions into one weighted entity graph.

Nodes are case-normalised entities carrying their surface variants and type
votes; edges combine explicit relations (weight 1 each) with same-chunk
co-mentions (weight 0.25 each), so a stated link outweighs a bare co-mention
four to one PER EDGE. That is a statement about weighting, not about the graph
that results: measured on the demo corpus, 92.8% of edges (51,176 of 55,146)
are co-mention only and just 3,970 carry an extracted relation. The graph is
mostly adjacency, and any claim resting on its relations should say so.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx

from .models import ChunkExtraction

RELATION_WEIGHT = 1.0
CO_MENTION_WEIGHT = 0.25


def _norm(name: str) -> str:
    return " ".join(name.split()).lower()


def build_graph(extractions: list[ChunkExtraction]) -> nx.Graph[str]:
    g: nx.Graph[str] = nx.Graph()
    for ext in extractions:
        norms: list[str] = []
        for ent in ext.entities:
            key = _norm(ent.name)
            if not g.has_node(key):
                g.add_node(key, variants=Counter(), types=Counter(), chunk_ids=set())
            node = g.nodes[key]
            node["variants"][ent.name] += 1
            node["types"][ent.type] += 1
            node["chunk_ids"].add(ext.chunk_id)
            norms.append(key)
        for a, b in combinations(sorted(set(norms)), 2):
            _bump(g, a, b, CO_MENTION_WEIGHT, None)
        for rel in ext.relations:
            src, tgt = _norm(rel.source), _norm(rel.target)
            if src != tgt and g.has_node(src) and g.has_node(tgt):
                _bump(g, src, tgt, RELATION_WEIGHT, rel.predicate)
    return g


def _bump(g: nx.Graph[str], a: str, b: str, weight: float, predicate: str | None) -> None:
    if g.has_edge(a, b):
        g.edges[a, b]["weight"] += weight
    else:
        g.add_edge(a, b, weight=weight, predicates=Counter())
    if predicate:
        g.edges[a, b]["predicates"][predicate] += 1


def _top(counter: Counter[str]) -> str:
    """Deterministic Counter winner: count desc, then lexicographic — never
    insertion order, which differs between a built and a JSON-loaded graph."""
    return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def display_name(g: nx.Graph[str], node: str) -> str:
    return _top(g.nodes[node]["variants"])


def node_type(g: nx.Graph[str], node: str) -> str:
    return _top(g.nodes[node]["types"])


def stats(g: nx.Graph[str]) -> dict[str, int]:
    return {"nodes": g.number_of_nodes(), "edges": g.number_of_edges()}


def save_graph(g: nx.Graph[str], path: Path) -> None:
    import json

    data = nx.node_link_data(g, edges="edges")
    for node in data["nodes"]:
        node["variants"] = dict(node["variants"])
        node["types"] = dict(node["types"])
        node["chunk_ids"] = sorted(node["chunk_ids"])
    for edge in data["edges"]:
        edge["predicates"] = dict(edge.get("predicates") or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=0, sort_keys=True) + "\n")


def load_graph(path: Path) -> nx.Graph[str]:
    import json

    data = json.loads(path.read_text())
    g: nx.Graph[str] = nx.node_link_graph(data, edges="edges")
    for _, attrs in g.nodes(data=True):
        attrs["variants"] = Counter(attrs["variants"])
        attrs["types"] = Counter(attrs["types"])
        attrs["chunk_ids"] = set(attrs["chunk_ids"])
    for _, _, attrs in g.edges(data=True):
        attrs["predicates"] = Counter(attrs["predicates"])
    return g


def to_igraph(g: nx.Graph[str]) -> tuple[Any, list[str]]:
    """networkx -> igraph with a stable node order (for Leiden)."""
    import igraph

    nodes = sorted(g.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    edges = [(index[a], index[b]) for a, b in g.edges()]
    weights = [g.edges[a, b]["weight"] for a, b in g.edges()]
    ig = igraph.Graph(n=len(nodes), edges=edges)
    ig.es["weight"] = weights
    return ig, nodes
