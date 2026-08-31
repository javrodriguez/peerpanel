"""Retrieval rung tests — all deterministic; the exclusion law swept over
EVERY mode (the invariant-sweep rule: an absolute is tested everywhere it
can be violated, not just where it is expected to hold)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from peerpanel.graph.build import build_graph
from peerpanel.graph.models import ChunkExtraction, Entity, Relation
from peerpanel.graph.summaries import CommunityReport
from peerpanel.providers.base import ChatResponse
from peerpanel.retrieval import (
    BM25Retriever,
    GraphGlobalRetriever,
    GraphLocalRetriever,
    Index,
    VectorRetriever,
    rrf,
)
from peerpanel.text.chunks import Chunk

CHUNKS = [
    Chunk("docA:0:aaaa0000", "docA", 0, "Met4 activates sulfur metabolism in budding yeast."),
    Chunk("docB:0:bbbb0000", "docB", 0, "RNA sequencing measures transcript abundance."),
    Chunk("docTwin:0:cccc0000", "docTwin", 0, "Met4 and sulfur metabolism in the twin paper."),
]

VECTORS = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]], dtype=np.float32
)


class _StubEmbed:
    name = "stub"
    dim = 3

    def __init__(self, vec: list[float]) -> None:
        self._vec = np.asarray([vec], dtype=np.float32)

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        return np.repeat(self._vec, len(texts), axis=0)


def _graph() -> object:
    return build_graph([
        ChunkExtraction(
            chunk_id="docA:0:aaaa0000",
            entities=[Entity(name="Met4", type="gene_or_protein"),
                      Entity(name="sulfur metabolism", type="process_or_phenotype")],
            relations=[Relation(source="Met4", predicate="activates", target="sulfur metabolism")],
        ),
        ChunkExtraction(
            chunk_id="docTwin:0:cccc0000",
            entities=[Entity(name="Met4", type="gene_or_protein")],
            relations=[],
        ),
        ChunkExtraction(
            chunk_id="docB:0:bbbb0000",
            entities=[Entity(name="RNA sequencing", type="method_or_tool")],
            relations=[],
        ),
    ])


def _reports() -> list[CommunityReport]:
    return [
        CommunityReport(community_id=0, resolution=1.0, size=2,
                        member_names=["Met4", "sulfur metabolism"],
                        title="Sulfur regulation", summary="Met4 drives sulfur pathways."),
        CommunityReport(community_id=1, resolution=1.0, size=1,
                        member_names=["RNA sequencing"],
                        title="Sequencing methods", summary="Transcript measurement methods."),
    ]


def _assignment() -> dict[str, int]:
    return {"met4": 0, "sulfur metabolism": 0, "rna sequencing": 1}


def _all_modes(index: Index) -> list[object]:
    graph = _graph()
    return [
        BM25Retriever(index),
        VectorRetriever(index, _StubEmbed([1.0, 0.0, 0.0])),
        GraphLocalRetriever(index, graph, embedder=None),  # type: ignore[arg-type]
        GraphGlobalRetriever(index, graph, _reports(), _assignment()),  # type: ignore[arg-type]
    ]


class TestExclusionLawEveryMode:
    def test_twin_chunks_never_surface_in_any_mode(self) -> None:
        index = Index.build(CHUNKS, VECTORS, exclude_docs={"docTwin"})
        assert index.excluded_docs == {"docTwin"}  # non-empty, recorded
        assert index.dropped_chunk_count == 1
        for retriever in _all_modes(index):
            hits = retriever.search("Met4 sulfur metabolism", k=10)  # type: ignore[attr-defined]
            assert all(h.doc_id != "docTwin" for h in hits), type(retriever).__name__

    def test_absent_twin_means_empty_exclusion_record(self) -> None:
        index = Index.build(CHUNKS[:2], VECTORS[:2], exclude_docs={"docTwin"})
        assert index.excluded_docs == set()  # the vacuous state IS detectable


class TestRungs:
    def test_bm25_finds_lexical_match(self) -> None:
        index = Index.build(CHUNKS, exclude_docs={"docTwin"})
        hits = BM25Retriever(index).search("sulfur metabolism yeast", k=2)
        assert hits[0].doc_id == "docA"

    def test_vector_follows_the_embedding(self) -> None:
        index = Index.build(CHUNKS, VECTORS, exclude_docs={"docTwin"})
        hits = VectorRetriever(index, _StubEmbed([0.0, 1.0, 0.0])).search("anything", k=1)
        assert hits[0].doc_id == "docB"

    def test_rrf_rewards_agreement(self) -> None:
        a = BM25Retriever(Index.build(CHUNKS)).search("Met4 sulfur", k=3)
        b = VectorRetriever(
            Index.build(CHUNKS, VECTORS), _StubEmbed([1.0, 0.0, 0.0])
        ).search("q", k=3)
        fused = rrf([a, b], k=3)
        assert fused[0].chunk_id == "docA:0:aaaa0000"

    def test_graph_local_votes_through_entities(self) -> None:
        index = Index.build(CHUNKS, exclude_docs={"docTwin"})
        hits = GraphLocalRetriever(index, _graph()).search("What does Met4 regulate?", k=3)  # type: ignore[arg-type]
        assert hits and hits[0].doc_id == "docA"

    def test_graph_global_routes_via_reports(self) -> None:
        index = Index.build(CHUNKS, exclude_docs={"docTwin"})
        retriever = GraphGlobalRetriever(index, _graph(), _reports(), _assignment())  # type: ignore[arg-type]
        hits = retriever.search("how are sulfur pathways regulated", k=3)
        assert hits and hits[0].doc_id == "docA"
        method_hits = retriever.search("transcript measurement sequencing", k=3)
        assert method_hits and method_hits[0].doc_id == "docB"


class _StubChat:
    name = "stub-chat"

    def chat(self, *, system: str, user: str, json_schema: object = None,
             temperature: float = 0.0, max_tokens: int = 2048) -> ChatResponse:
        return ChatResponse(
            text="Sulfur pathways are driven by Met4 [community 0].",
            model="stub", prompt_tokens=1, completion_tokens=1,
        )


class TestGlobalAnswer:
    def test_answer_cites_communities(self) -> None:
        index = Index.build(CHUNKS, exclude_docs={"docTwin"})
        retriever = GraphGlobalRetriever(index, _graph(), _reports(), _assignment())  # type: ignore[arg-type]
        out = retriever.answer("how are sulfur pathways regulated", _StubChat())
        assert out.community_ids == [0]
        assert "Met4" in out.answer
