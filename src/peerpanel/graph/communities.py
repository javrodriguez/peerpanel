"""Multi-level Leiden community detection (seeded — identical runs partition
identically), the structural half of what makes this GraphRAG rather than
plain knowledge-graph RAG: communities get summarised, and global search
map-reduces over those summaries."""

from __future__ import annotations

import networkx as nx

SEED = 42
RESOLUTIONS: tuple[float, ...] = (0.3, 1.0, 3.0)  # coarse -> fine


def detect(
    g: nx.Graph[str], resolutions: tuple[float, ...] = RESOLUTIONS
) -> dict[float, dict[str, int]]:
    """Leiden at each resolution; returns {resolution: {node: community_id}}."""
    import leidenalg

    from .build import to_igraph

    if g.number_of_nodes() == 0:
        return {r: {} for r in resolutions}
    ig, nodes = to_igraph(g)
    out: dict[float, dict[str, int]] = {}
    for resolution in resolutions:
        partition = leidenalg.find_partition(
            ig,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=SEED,
        )
        out[resolution] = {nodes[i]: cid for i, cid in enumerate(partition.membership)}
    return out


def members(assignment: dict[str, int]) -> dict[int, list[str]]:
    """Invert a node->community map into community -> sorted member list."""
    inverted: dict[int, list[str]] = {}
    for node, cid in assignment.items():
        inverted.setdefault(cid, []).append(node)
    return {cid: sorted(nodes) for cid, nodes in inverted.items()}
