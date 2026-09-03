"""Extraction tests — candidates are real spaCy in CI; typing via unit stub;
the live LLM road carries the ollama marker."""

from __future__ import annotations

from pathlib import Path

import pytest

from peerpanel.graph.extract import (
    EXTRACTION_SCHEMA,
    candidate_terms,
    extract_chunk,
    extract_many,
)
from peerpanel.graph.models import ENTITY_TYPES
from peerpanel.providers.base import ChatResponse
from peerpanel.text.chunks import Chunk

BIO = (
    "Met4 activates sulfur metabolism in Saccharomyces cerevisiae. "
    "The transcription factor binds Cbf1 near MET17 promoters."
)


class TestCandidates:
    def test_bio_terms_found(self) -> None:
        terms = candidate_terms(BIO)
        joined = " | ".join(terms).lower()
        assert "met4" in joined
        assert "sulfur metabolism" in joined
        assert "saccharomyces cerevisiae" in joined

    def test_determinism_and_dedupe(self) -> None:
        assert candidate_terms(BIO) == candidate_terms(BIO)
        twice = candidate_terms(BIO + " " + BIO)
        assert len(twice) == len({t.lower() for t in twice})

    def test_junk_filtered(self) -> None:
        terms = candidate_terms("It was 42. He saw 3.14 there.")
        assert all(len(t) >= 3 and any(c.isalpha() for c in t) for t in terms)

    def test_casing_preserved(self) -> None:
        assert any(t == "Met4" for t in candidate_terms(BIO))


class _StubChat:
    """Unit stub of our own seam; the live road is ollama-marked below.

    payloads: successive responses (repeats the last one when exhausted).
    """

    name = "stub-chat"

    def __init__(self, *payloads: str) -> None:
        self.payloads = list(payloads)
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
        self.calls.append(
            {"user": user, "schema": json_schema, "temp": temperature, "max_tokens": max_tokens}
        )
        payload = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        return ChatResponse(text=payload, model="stub", prompt_tokens=1, completion_tokens=1)


GOOD = (
    '{"entities": [{"name": "Met4", "type": "gene_or_protein"},'
    ' {"name": "sulfur metabolism", "type": "process_or_phenotype"}],'
    ' "relations": [{"source": "Met4", "predicate": "activates", "target": "sulfur metabolism"},'
    ' {"source": "Met4", "predicate": "binds", "target": "Cbf1"}]}'
)
TRUNCATED = '{"entities": [{"name": "Met4", "ty'


def _chunk(cid: str = "d:0:abcd1234") -> Chunk:
    return Chunk(cid, "d", 0, BIO)


class TestTyping:
    def test_prompt_carries_candidates_and_schema_at_temp0(self) -> None:
        stub = _StubChat(GOOD)
        extract_chunk(_chunk(), stub)
        call = stub.calls[0]
        assert "Met4" in str(call["user"])
        assert call["schema"] == EXTRACTION_SCHEMA
        assert call["temp"] == 0.0

    def test_dangling_relation_dropped(self) -> None:
        out = extract_chunk(_chunk(), _StubChat(GOOD))
        assert [e.name for e in out.entities] == ["Met4", "sulfur metabolism"]
        assert len(out.relations) == 1
        assert out.relations[0].predicate == "activates"

    def test_unknown_entity_type_dropped(self) -> None:
        bad = '{"entities": [{"name": "X", "type": "martian"}], "relations": []}'
        out = extract_chunk(_chunk(), _StubChat(bad))
        assert out.entities == []

    def test_cache_prevents_second_call(self, tmp_path: Path) -> None:
        stub = _StubChat(GOOD)
        first = extract_chunk(_chunk(), stub, cache_dir=tmp_path)
        again = extract_chunk(_chunk(), stub, cache_dir=tmp_path)
        assert len(stub.calls) == 1
        assert first == again

    def test_cache_key_varies_by_provider(self, tmp_path: Path) -> None:
        a, b = _StubChat(GOOD), _StubChat(GOOD)
        b.name = "other-provider"
        extract_chunk(_chunk(), a, cache_dir=tmp_path)
        extract_chunk(_chunk(), b, cache_dir=tmp_path)
        assert len(a.calls) == 1
        assert len(b.calls) == 1


class TestTruncationHandling:
    def test_retry_doubles_budget_then_succeeds(self) -> None:
        stub = _StubChat(TRUNCATED, GOOD)
        out = extract_chunk(_chunk(), stub, max_tokens=1000)
        assert len(stub.calls) == 2
        assert stub.calls[0]["max_tokens"] == 1000
        assert stub.calls[1]["max_tokens"] == 2000
        assert not out.truncated
        assert out.entities

    def test_double_failure_is_honest_and_cached(self, tmp_path: Path) -> None:
        stub = _StubChat(TRUNCATED)
        out = extract_chunk(_chunk(), stub, cache_dir=tmp_path)
        assert out.truncated
        assert out.entities == [] and out.relations == []
        assert len(stub.calls) == 2
        again = extract_chunk(_chunk(), stub, cache_dir=tmp_path)
        assert again.truncated
        assert len(stub.calls) == 2  # cached — the budget is not re-burned

    def test_non_dict_json_treated_as_failure(self) -> None:
        out = extract_chunk(_chunk(), _StubChat('["a list"]'))
        assert out.truncated


class TestExtractMany:
    def test_order_preserved_across_pool(self, tmp_path: Path) -> None:
        stub = _StubChat(GOOD)
        chunks = [_chunk(f"d:{i}:aaaa000{i}") for i in range(6)]
        outs = extract_many(chunks, stub, cache_dir=tmp_path, concurrency=3)
        assert [o.chunk_id for o in outs] == [c.chunk_id for c in chunks]
        assert len(stub.calls) == 6


@pytest.mark.ollama
class TestLiveExtraction:
    def test_real_typing_on_llama(self) -> None:
        from peerpanel.providers import OllamaOpenAIChat

        out = extract_chunk(_chunk("live:0:ffff0000"), OllamaOpenAIChat("llama3.1:8b"))
        assert out.entities
        assert all(e.type in ENTITY_TYPES for e in out.entities)
        names = " ".join(e.name for e in out.entities).lower()
        assert "met4" in names or "sulfur" in names
