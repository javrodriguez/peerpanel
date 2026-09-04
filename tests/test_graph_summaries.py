"""Community-report tests — grounded prompts, caching, honest failure."""

from __future__ import annotations

from pathlib import Path

import pytest

from peerpanel.agents.schemas import CallStats
from peerpanel.graph.build import build_graph
from peerpanel.graph.models import ChunkExtraction, Entity, Relation
from peerpanel.graph.summaries import (
    MIN_MEMBERS,
    REPORT_SCHEMA,
    summarise_communities,
)
from peerpanel.providers.base import ChatResponse, TokenLedger


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

    def chat(
        self,
        *,
        system: str,
        user: str,
        json_schema: object = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse:
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


class TestTheRunRecord:
    """The layer that shipped no run record until round 3 found it (report-2, 3b).

    The community reports are the only input to graphrag-global's ranking — the rung
    this repository reports as its winner — and they ride the one wire that cannot set
    its own window. `--publish` writes `results/summaries-stats-<corpus>.json` from the
    ledger threaded through here, so what the ledger counts IS the published number:
    one call per report this run generated, and nothing at all for a report served from
    the committed cache. A warm run recording calls would be a record claiming work it
    did not do; a cold run recording none would be a layer still invisible.
    """

    def test_the_ledger_counts_one_call_per_generated_report(self, tmp_path: Path) -> None:
        g, assignment = _graph_and_assignment()
        stub = _StubChat(GOOD)
        ledger = TokenLedger()
        reports = summarise_communities(
            g,  # type: ignore[arg-type]
            assignment,
            1.0,
            stub,  # type: ignore[arg-type]
            cache_dir=tmp_path,
            ledger=ledger,
        )
        assert reports, "nothing was summarised, so this proves nothing about the record"
        assert (ledger.calls, len(stub.calls)) == (len(reports), len(reports))
        # The published record is CallStats.from_ledger(ledger) (__main__, `--publish`),
        # so bind the number a reader sees, not only the counter behind it.
        assert CallStats.from_ledger(ledger).calls == len(reports)
        assert ledger.prompt_tokens == len(reports)  # the stub reports one per call

    def test_a_replayed_report_records_no_call(self, tmp_path: Path) -> None:
        g, assignment = _graph_and_assignment()
        stub = _StubChat(GOOD)
        cold = TokenLedger()
        first = summarise_communities(
            g,  # type: ignore[arg-type]
            assignment,
            1.0,
            stub,  # type: ignore[arg-type]
            cache_dir=tmp_path,
            ledger=cold,
        )
        warm = TokenLedger()
        second = summarise_communities(
            g,  # type: ignore[arg-type]
            assignment,
            1.0,
            stub,  # type: ignore[arg-type]
            cache_dir=tmp_path,
            ledger=warm,
        )
        assert [r.summary for r in second] == [r.summary for r in first]
        assert cold.calls == len(first) and len(stub.calls) == len(first)
        assert warm.calls == 0, (
            "the second pass was served entirely from the cache and called nothing; a "
            "record that counted calls here would describe a run that did not happen"
        )
        assert CallStats.from_ledger(warm).window_sources == [], (
            "a run that made no call read no window, so it names no source for one"
        )


@pytest.mark.ollama
class TestLiveSummary:
    def test_real_report_on_llama(self) -> None:
        from peerpanel.providers import OllamaOpenAIChat

        g, assignment = _graph_and_assignment()
        report = summarise_communities(g, assignment, 1.0, OllamaOpenAIChat("llama3.1:8b"))[0]
        assert report.title
        assert len(report.summary.split()) > 5
