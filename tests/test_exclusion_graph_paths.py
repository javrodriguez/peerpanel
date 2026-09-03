"""The self-exclusion law, swept over the GRAPH-MEDIATED paths.

Chunk-level filtering stops excluded text from being read. It does not, on its
own, stop an excluded document's ENTITIES from steering which surviving chunks
rank — the twin's authors and coined terms would still match a query and vote
for their neighbours. These tests pin the stronger law: an entity evidenced
only by excluded documents does not exist for that run.
"""

from __future__ import annotations

from peerpanel.graph.build import build_graph
from peerpanel.graph.models import ChunkExtraction, Entity, Relation
from peerpanel.graph.summaries import CommunityReport
from peerpanel.retrieval import GraphGlobalRetriever, GraphLocalRetriever, Index
from peerpanel.text.chunks import Chunk

CHUNKS = [
    Chunk("keep:0:aaaa0000", "keep", 0, "Sulfur metabolism proceeds through sulfate reduction."),
    Chunk("twin:0:bbbb0000", "twin", 0, "Hollingsworth measured sulfur metabolism in detail."),
]

EXTRACTIONS = [
    ChunkExtraction(
        chunk_id="keep:0:aaaa0000",
        entities=[
            Entity(name="sulfur metabolism", type="process_or_phenotype"),
            Entity(name="sulfate reduction", type="process_or_phenotype"),
        ],
        relations=[
            Relation(
                source="sulfur metabolism", predicate="proceeds via", target="sulfate reduction"
            )
        ],
    ),
    ChunkExtraction(
        chunk_id="twin:0:bbbb0000",
        entities=[
            # Shared with the surviving chunk — survives, but see the residual note.
            Entity(name="sulfur metabolism", type="process_or_phenotype"),
            # Evidenced ONLY by the excluded document — must not exist for this run.
            Entity(name="Hollingsworth", type="other"),
        ],
        relations=[
            Relation(source="Hollingsworth", predicate="measured", target="sulfur metabolism")
        ],
    ),
]

REPORTS = [
    CommunityReport(
        community_id=0,
        resolution=1.0,
        size=3,
        member_names=["sulfur metabolism", "sulfate reduction", "Hollingsworth"],
        title="Sulfur metabolism",
        summary="Sulfur metabolism and its measurement.",
    )
]
ASSIGNMENT = {"sulfur metabolism": 0, "sulfate reduction": 0, "hollingsworth": 0}


def _graph() -> object:
    return build_graph(EXTRACTIONS)


class TestLiveNodes:
    def test_twin_only_entities_are_not_live(self) -> None:
        index = Index.build(CHUNKS, exclude_docs={"twin"})
        live = index.live_nodes(_graph())  # type: ignore[arg-type]
        assert "hollingsworth" not in live
        assert {"sulfur metabolism", "sulfate reduction"} <= live

    def test_no_exclusion_means_every_node_live(self) -> None:
        index = Index.build(CHUNKS)
        assert index.live_nodes(_graph()) == set(_graph().nodes())  # type: ignore[arg-type]


class TestGraphPathsCannotRouteThroughExcludedEntities:
    def test_local_search_ignores_a_twin_only_entity(self) -> None:
        index = Index.build(CHUNKS, exclude_docs={"twin"})
        retriever = GraphLocalRetriever(index, _graph(), embedder=None)  # type: ignore[arg-type]
        # Querying the twin's own coined term must not route into the graph at all.
        assert retriever.search("Hollingsworth", k=5) == []
        # And no surviving chunk may be ranked BY that entity's vote.
        assert "hollingsworth" not in retriever._matched_nodes("Hollingsworth measured")

    def test_local_search_still_works_through_surviving_entities(self) -> None:
        index = Index.build(CHUNKS, exclude_docs={"twin"})
        retriever = GraphLocalRetriever(index, _graph(), embedder=None)  # type: ignore[arg-type]
        hits = retriever.search("sulfate reduction", k=5)
        assert hits and all(h.doc_id == "keep" for h in hits)

    def test_global_search_drops_excluded_members_from_community_expansion(self) -> None:
        index = Index.build(CHUNKS, exclude_docs={"twin"})
        retriever = GraphGlobalRetriever(index, _graph(), REPORTS, ASSIGNMENT)  # type: ignore[arg-type]
        hits = retriever.search("sulfur metabolism", k=5)
        assert all(h.doc_id == "keep" for h in hits)
        assert "hollingsworth" not in retriever._live

    def test_chunk_level_filter_still_holds_everywhere(self) -> None:
        index = Index.build(CHUNKS, exclude_docs={"twin"})
        graph = _graph()
        for retriever in (
            GraphLocalRetriever(index, graph, embedder=None),  # type: ignore[arg-type]
            GraphGlobalRetriever(index, graph, REPORTS, ASSIGNMENT),  # type: ignore[arg-type]
        ):
            for query in ("sulfur metabolism", "Hollingsworth", "sulfate reduction"):
                assert all(h.doc_id != "twin" for h in retriever.search(query, k=10))
