"""Community-report tests — grounded prompts, caching, honest failure."""

from __future__ import annotations

from pathlib import Path

import pytest

from peerpanel.graph.build import build_graph
from peerpanel.graph.models import ChunkExtraction, Entity, Relation
from peerpanel.graph.summaries import (
    MIN_MEMBERS,
    REPORT_SCHEMA,
    summarise_communities,
)
from peerpanel.providers.base import ChatResponse


def _graph_and_assignment() -> tuple[object, dict[str, int]]:
    ext = ChunkExtraction(
        chunk_id="c1",
        entities=[
            Entity(name="Met4", type="gene_or_protein"),
            Entity(name="Cbf1", type="gene_or_protein"),
            Entity(name="sulfur metabolism", type="process_or_phenotype"),
            Entity(name="lonely", type="other"),
        ],
        relations=[
            Relation(source="Met4", predicate="activates", target="sulfur metabolism"),
            Relation(source="Met4", predicate="binds", target="Cbf1"),
        ],
    )
    g = build_graph([ext])
    assignment = {"met4": 0, "cbf1": 0, "sulfur metabolism": 0, "lonely": 1}
    return g, assignment


class _StubChat:
    name = "stub-chat"

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    def chat(self, *, system: str, user: str, json_schema: object = None,
             temperature: float = 0.0, max_tokens: int = 2048) -> ChatResponse:
        self.calls.append({"user": user, "schema": json_schema})
        return ChatResponse(text=self.payload, model="stub", prompt_tokens=1, completion_tokens=1)


GOOD = '{"title": "Sulfur regulation", "summary": "Met4 activates sulfur metabolism with Cbf1."}'


class TestSummaries:
    def test_small_communities_skipped(self) -> None:
        g, assignment = _graph_and_assignment()
        stub = _StubChat(GOOD)
        reports = summarise_communities(g, assignment, 1.0, stub)  # type: ignore[arg-type]
        assert len(reports) == 1  # the singleton "lonely" community is below MIN_MEMBERS
        assert reports[0].size == 3 >= MIN_MEMBERS

    def test_prompt_grounded_in_entities_and_relations(self) -> None:
        g, assignment = _graph_and_assignment()
        stub = _StubChat(GOOD)
        summarise_communities(g, assignment, 1.0, stub)  # type: ignore[arg-type]
        user = str(stub.calls[0]["user"])
        assert "Met4 (gene_or_protein)" in user
        assert "—activates→" in user
        assert stub.calls[0]["schema"] == REPORT_SCHEMA

    def test_report_carries_members_and_text(self) -> None:
        g, assignment = _graph_and_assignment()
        report = summarise_communities(g, assignment, 1.0, _StubChat(GOOD))[0]  # type: ignore[arg-type]
        assert report.title == "Sulfur regulation"
        assert "Met4" in report.member_names
        assert not report.truncated

    def test_cache_hit_skips_call_and_rebinds_ids(self, tmp_path: Path) -> None:
        g, assignment = _graph_and_assignment()
        stub = _StubChat(GOOD)
        first = summarise_communities(g, assignment, 1.0, stub, cache_dir=tmp_path)  # type: ignore[arg-type]
        shifted = {n: cid + 7 for n, cid in assignment.items()}
        second = summarise_communities(g, shifted, 0.3, stub, cache_dir=tmp_path)  # type: ignore[arg-type]
        assert len(stub.calls) == 1  # same content -> cached despite new ids
        assert second[0].community_id == 7
        assert second[0].resolution == 0.3
        assert second[0].summary == first[0].summary

    def test_invalid_json_is_honest(self) -> None:
        g, assignment = _graph_and_assignment()
        report = summarise_communities(g, assignment, 1.0, _StubChat('{"tit'))[0]  # type: ignore[arg-type]
        assert report.truncated
        assert report.summary == ""


@pytest.mark.ollama
class TestLiveSummary:
    def test_real_report_on_llama(self) -> None:
        from peerpanel.providers import OllamaOpenAIChat

        g, assignment = _graph_and_assignment()
        report = summarise_communities(g, assignment, 1.0, OllamaOpenAIChat("llama3.1:8b"))[0]
        assert report.title
        assert len(report.summary.split()) > 5
