"""The context-window contract (providers/context.py, DECISIONS.md D19).

A prompt is never silently cut: the native wire sizes the window per call, the
OpenAI wire reads the window its own calls run in and refuses what will not
fit, and both wires refuse an answer whose prompt count carries the truncation
signature. Every test here is model-free — the wires are driven through stub
clients that record what would have gone on the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import pytest

from peerpanel.agents.json_call import call_json, parse_object
from peerpanel.agents.schemas import CallStats, PanelReview
from peerpanel.graph.extract import EXTRACTION_MODEL, EXTRACTION_PROVIDER_NAME
from peerpanel.graph.summaries import SUMMARY_MODEL, SUMMARY_PROVIDER_NAME
from peerpanel.providers import ChatResponse, OllamaNativeChat, OllamaOpenAIChat, TokenLedger
from peerpanel.providers.context import (
    CHARS_PER_TOKEN_SIGNATURE,
    CHARS_PER_TOKEN_SIZING,
    CONTEXT_CEILING,
    CONTEXT_FLOOR,
    TEMPLATE_OVERHEAD_TOKENS,
    ContextTooSmall,
    PromptTooLarge,
    PromptTruncated,
    assert_fits,
    check_not_truncated,
    context_for_call,
    context_needed,
    tokens_upper_bound,
)
from peerpanel.text.chunks import MIN_TAIL_WORDS, _sentences, chunk_document, split_oversize

# Measured on the 15,000-character reviewer prompt (context.py docstring).
# chars / prompt_eval_count, measured on the real prompts with num_predict=1. The
# extraction entries are the ones that matter: a passage plus a JSON candidate list
# tokenises far worse than prose, and the sizing bound has to hold for the worst.
MEASURED_CHARS_PER_TOKEN = {
    "reviewer qwen2:7b": 3.49,
    "reviewer llama3.1:8b": 3.90,
    "extraction llama3.1:8b (thinnest)": 4.40,
    "extraction llama3.1:8b (median)": 3.85,
    "extraction llama3.1:8b (worst in ci)": 2.71,
    "extraction llama3.1:8b (worst in demo)": 2.21,
}
# What Ollama reports for a prompt cut to a 4,096 window's half (llm/llama_server.go).
TRUNCATED_COUNT_ON_4096 = 2050


def _prompt(chars: int) -> tuple[str, str]:
    system = "You are a reviewer. " * 10
    return system, "x" * max(0, chars - len(system))


class TestSizingBound:
    @pytest.mark.parametrize("model,ratio", sorted(MEASURED_CHARS_PER_TOKEN.items()))
    def test_upper_bound_exceeds_every_measured_tokenizer(self, model: str, ratio: float) -> None:
        chars = 15_000
        true_tokens = chars / ratio
        assert tokens_upper_bound(chars) > true_tokens, model
        # Including the worst: a bound above any measured ratio would under-count the
        # prompt class this project actually sends most of.
        assert min(MEASURED_CHARS_PER_TOKEN.values()) > CHARS_PER_TOKEN_SIZING

    def test_the_bound_holds_for_the_largest_prompt_each_window_admits(self) -> None:
        """The margin, stated as a number: at what true ratio would sizing fail?"""
        for window in (4096, 8192, 16384):
            budget = 1024
            admitted_chars = (
                window - TEMPLATE_OVERHEAD_TOKENS - budget - 1
            ) * CHARS_PER_TOKEN_SIZING
            breaking_ratio = admitted_chars / (window - budget)
            assert breaking_ratio < min(MEASURED_CHARS_PER_TOKEN.values()), (
                f"window {window}: sizing fails at {breaking_ratio:.2f} chars/token, "
                f"inside the measured range"
            )

    def test_overhead_is_added_even_to_an_empty_prompt(self) -> None:
        assert tokens_upper_bound(0) == TEMPLATE_OVERHEAD_TOKENS

    def test_windows_are_powers_of_two_from_the_floor(self) -> None:
        seen = set()
        refused = 0
        for chars in range(0, 90_000, 997):
            system, user = _prompt(chars)
            try:
                window = context_needed(system, user, 1024)
            except PromptTooLarge:
                refused += 1  # above the ceiling: a window, not a bigger window
                continue
            assert window >= CONTEXT_FLOOR
            assert window & (window - 1) == 0, window  # a power of two
            assert window <= CONTEXT_CEILING, (
                f"{chars:,} chars was given a {window:,}-token window, above the "
                f"{CONTEXT_CEILING:,} ceiling — no server here can honour it"
            )
            assert window >= tokens_upper_bound(chars) + 1024 + 1
            assert window // 2 < tokens_upper_bound(chars) + 1024 + 1 or window == CONTEXT_FLOOR
            seen.add(window)
        assert {4096, 8192, 16384, 32768} <= seen
        assert refused, "the sweep never reached the ceiling, so its refusal is untested"

    def test_the_reviewer_prompt_does_not_fit_the_default_window(self) -> None:
        # The defect D19 names: ~15,000 characters plus a 1,024-token budget, against a
        # 4,096-token default. What the window grows TO is arithmetic that has moved with
        # the bound; that it must grow at all is the claim.
        system, user = _prompt(15_000)
        assert context_needed(system, user, 1024) > CONTEXT_FLOOR
        assert context_for_call(system, user, 1024, model="qwen2:7b") == 16384

    def test_the_output_budget_counts_toward_the_window(self) -> None:
        """The answer has to fit beside the question: the same prompt needs a bigger
        window when it is allowed to say more."""
        system, user = _prompt(5_000)
        assert context_needed(system, user, 64) == CONTEXT_FLOOR
        assert context_needed(system, user, 1024) == 2 * CONTEXT_FLOOR

    def test_the_raw_computation_refuses_the_ceiling_too(self) -> None:
        """The ceiling is a rule, so it lives where the window is computed. This returned
        65,536 for a large enough prompt — a window no server here can be given, which
        the power-of-two sweep above accepted without complaint."""
        system, user = _prompt(60_000)
        with pytest.raises(PromptTooLarge):
            context_needed(system, user, 1024)

    def test_the_window_is_a_window_and_not_a_token_count(self) -> None:
        """`context_for_call` returns what goes into `num_ctx`. Returning the raw need
        instead would silently serve every call in a window the size of its own prompt."""
        system, user = _prompt(15_000)
        assert context_for_call(system, user, 1024, model="qwen2:7b") == 16384
        assert context_for_call(system, user, 1024, model="qwen2:7b") == context_needed(
            system, user, 1024
        )

    def test_above_the_ceiling_is_refused_not_squeezed(self) -> None:
        system, user = _prompt(3 * CONTEXT_CEILING)
        with pytest.raises(PromptTooLarge) as err:
            context_for_call(system, user, 1024, model="llama3.1:8b")
        assert f"{CONTEXT_CEILING:,}" in str(err.value)
        assert "Budget the prompt at its source" in str(err.value)


class TestTruncationSignature:
    @pytest.mark.parametrize("model,ratio", sorted(MEASURED_CHARS_PER_TOKEN.items()))
    def test_a_genuine_count_never_trips_it(self, model: str, ratio: float) -> None:
        for chars in (500, 4_000, 15_000, 40_000):
            check_not_truncated(int(chars / ratio), chars, model=model, context=32768)

    def test_a_count_cut_to_the_windows_half_trips_it(self) -> None:
        # 15,000 characters is ~4,100 tokens on qwen2; a 4,096 window hands back 2,050.
        with pytest.raises(PromptTruncated) as err:
            check_not_truncated(TRUNCATED_COUNT_ON_4096, 15_000, model="qwen2:7b", context=4096)
        message = str(err.value)
        assert "2,050 prompt tokens" in message and "15,000-character" in message
        assert "4,096-token window" in message
        assert "native wire" in message.lower() and "openai wire" in message.lower()

    def test_no_measured_prompt_can_false_trip_the_signature(self) -> None:
        assert max(MEASURED_CHARS_PER_TOKEN.values()) < CHARS_PER_TOKEN_SIGNATURE

    def test_the_signature_s_blind_spot_is_where_the_docstring_says_it_is(self) -> None:
        """The check catches a served window far smaller than the one asked for. It does
        NOT catch a prompt that overruns the window it was correctly given, and this test
        exists to keep that limit visible rather than assumed away.

        A cut prompt reports about half the window, so it reads at roughly twice its true
        ratio. On the densest prompt class that is 2 x 2.71 = 5.42 — under the threshold,
        so such a cut would pass unnoticed. That case is prevented by the sizing bound's
        margin (context.py), not detected here, and the threshold is not lowered to catch
        it because a sequence-heavy passage — this corpus is full of them — reads high and
        would then crash a healthy run.
        """
        worst_ratio = min(MEASURED_CHARS_PER_TOKEN.values())
        assert 2 * worst_ratio < CHARS_PER_TOKEN_SIGNATURE  # the blind spot, pinned
        # And the case it does catch: a prompt sized for 8,192 served in a 4,096 window.
        chars = int(8192 * worst_ratio)
        with pytest.raises(PromptTruncated):
            check_not_truncated(2050, chars, model="llama3.1:8b", context=4096)


class TestAssertFits:
    def test_refuses_a_fixed_window_that_cannot_hold_the_call(self) -> None:
        system, user = _prompt(15_000)
        with pytest.raises(ContextTooSmall) as err:
            assert_fits(4096, system, user, 1024, model="llama3.1:8b", remedy="DO THIS")
        assert "4,096-token window" in str(err.value)
        assert str(err.value).endswith("DO THIS")

    def test_passes_when_it_fits(self) -> None:
        system, user = _prompt(2_000)
        assert_fits(4096, system, user, 1024, model="llama3.1:8b", remedy="n/a")


# --- the wires, through stub clients ---------------------------------------------


@dataclass
class _NativeReply:
    content: str
    prompt_eval_count: int
    eval_count: int = 7

    @property
    def message(self) -> Any:
        return type("M", (), {"content": self.content})()


@dataclass
class _NativeStub:
    prompt_eval_count: int | None = None  # None: report a genuine count
    calls: list[dict[str, Any]] = field(default_factory=list)

    def chat(self, **kwargs: Any) -> _NativeReply:
        self.calls.append(kwargs)
        chars = sum(len(m["content"]) for m in kwargs["messages"])
        count = self.prompt_eval_count if self.prompt_eval_count is not None else chars // 4
        return _NativeReply('{"verdict": "ok"}', count)


class TestNativeWire:
    def test_sizes_the_window_per_call_and_records_it(self) -> None:
        provider = OllamaNativeChat("qwen2:7b")
        stub = _NativeStub()
        provider._client = stub  # type: ignore[assignment]
        system, user = _prompt(15_000)
        response = provider.chat(system=system, user=user, max_tokens=1024)
        (call,) = stub.calls
        assert call["options"]["num_ctx"] == 16384
        assert call["options"]["num_predict"] == 1024
        assert response.context == 16384
        assert response.prompt_tokens == 15_000 // 4
        small = _prompt(1_000)
        provider.chat(system=small[0], user=small[1], max_tokens=256)
        assert stub.calls[-1]["options"]["num_ctx"] == CONTEXT_FLOOR

    def test_refuses_the_truncation_signature(self) -> None:
        provider = OllamaNativeChat("qwen2:7b")
        provider._client = _NativeStub(prompt_eval_count=TRUNCATED_COUNT_ON_4096)  # type: ignore[assignment]
        system, user = _prompt(15_000)
        with pytest.raises(PromptTruncated):
            provider.chat(system=system, user=user, max_tokens=1024)

    def test_refuses_above_the_ceiling_before_any_call(self) -> None:
        provider = OllamaNativeChat("qwen2:7b")
        stub = _NativeStub()
        provider._client = stub  # type: ignore[assignment]
        system, user = _prompt(3 * CONTEXT_CEILING)
        with pytest.raises(PromptTooLarge):
            provider.chat(system=system, user=user, max_tokens=1024)
        assert stub.calls == []


@dataclass
class _Running:
    model: str
    context_length: int


@dataclass
class _PsStub:
    """The server as `ps` shows it: whatever window the last request loaded."""

    loaded: list[_Running]

    def ps(self) -> Any:
        return type("Ps", (), {"models": list(self.loaded)})()


@dataclass
class _OpenAIStub:
    """The OpenAI-compatible endpoint: every call reloads the runner at the server
    default (measured on Ollama 0.33 — see OllamaOpenAIChat.context)."""

    server_default: int
    ps: _PsStub
    prompt_tokens: int | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def chat(self) -> Any:
        stub = self

        class _Completions:
            def create(self, **kwargs: Any) -> Any:
                stub.calls.append(kwargs)
                stub.ps.loaded = [_Running(kwargs["model"], stub.server_default)]
                chars = sum(len(m["content"]) for m in kwargs["messages"])
                count = stub.prompt_tokens if stub.prompt_tokens is not None else chars // 4
                usage = type("U", (), {"prompt_tokens": count, "completion_tokens": 3})()
                choice = type("C", (), {"message": type("M", (), {"content": "{}"})()})()
                return type("R", (), {"usage": usage, "choices": [choice]})()

        return type("Chat", (), {"completions": _Completions()})()


def _openai_wire(server_default: int, resident: int | None = None, **kw: Any) -> tuple[Any, Any]:
    provider = OllamaOpenAIChat("llama3.1:8b")
    loaded = [] if resident is None else [_Running("llama3.1:8b", resident)]
    ps = _PsStub(loaded)
    client = _OpenAIStub(server_default, ps, **kw)
    provider._client = client  # type: ignore[assignment]
    provider._native = ps  # type: ignore[assignment]
    return provider, client


class TestOpenAIWire:
    def test_reads_the_window_its_own_calls_run_in_not_another_wires(self) -> None:
        # The native wire left the model resident at 8,192; this wire's next call
        # reloads it at the 4,096 default. A `ps` read taken first would say 8,192.
        provider, client = _openai_wire(server_default=4096, resident=8192)
        assert provider.context() == 4096
        assert client.calls[0]["max_tokens"] == 1  # the one-token warm-up went first
        assert provider.context() == 4096 and len(client.calls) == 1  # read once

    def test_refuses_a_call_the_default_window_cannot_hold(self) -> None:
        provider, client = _openai_wire(server_default=4096)
        system, user = _prompt(15_000)
        with pytest.raises(ContextTooSmall) as err:
            provider.chat(system=system, user=user, max_tokens=1024)
        assert "OLLAMA_CONTEXT_LENGTH" in str(err.value) and "native" in str(err.value)
        assert all(c["max_tokens"] == 1 for c in client.calls)  # nothing but the warm-up

    def test_serves_a_call_that_fits_and_records_the_window(self) -> None:
        provider, _ = _openai_wire(server_default=4096)
        system, user = _prompt(2_500)  # a community-report-sized prompt
        response = provider.chat(system=system, user=user, max_tokens=512)
        assert response.context == 4096 and response.prompt_tokens == 2_500 // 4

    def test_a_larger_server_default_admits_the_reviewer_prompt(self) -> None:
        provider, _ = _openai_wire(server_default=16384)
        system, user = _prompt(15_000)
        assert provider.chat(system=system, user=user, max_tokens=1024).context == 16384

    def test_refuses_the_truncation_signature_after_the_call(self) -> None:
        provider, _ = _openai_wire(server_default=16384, prompt_tokens=TRUNCATED_COUNT_ON_4096)
        system, user = _prompt(15_000)
        with pytest.raises(PromptTruncated):
            provider.chat(system=system, user=user, max_tokens=1024)


# --- the shared JSON call ---------------------------------------------------------


class _ScriptedProvider:
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.budgets: list[int] = []

    def chat(self, **kwargs: Any) -> ChatResponse:
        self.budgets.append(kwargs["max_tokens"])
        text = self.replies.pop(0)
        return ChatResponse(
            text=text, model="m", prompt_tokens=100, completion_tokens=len(text), context=4096
        )


class TestCallJson:
    def test_one_retry_at_double_budget_and_both_calls_ledgered(self) -> None:
        provider = _ScriptedProvider(['{"a": 1, "b": [', '{"a": 1}'])
        ledger = TokenLedger()
        data = call_json(
            provider, ledger, system="s", user="u", schema={"type": "object"}, max_tokens=300
        )
        assert data == {"a": 1}
        assert provider.budgets == [300, 600]
        assert ledger.calls == 2 and ledger.largest_prompt_tokens == 100
        assert ledger.smallest_context == 4096

    def test_two_failures_return_none_never_a_third_call(self) -> None:
        provider = _ScriptedProvider(["not json", "still not"])
        assert call_json(provider, None, system="s", user="u", schema={}, max_tokens=10) is None
        assert provider.budgets == [10, 20]

    def test_parse_object_is_strict_an_object_or_nothing(self) -> None:
        # Constrained decoding emits JSON and nothing else; prose around it means
        # the decoder was not constrained, and a bare list is not the schema's shape.
        assert parse_object('{"k": "v"}') == {"k": "v"}
        assert parse_object('noise {"k": "v"} trailing') is None
        assert parse_object("[1, 2]") is None
        assert parse_object("") is None


# --- the chunker: no oversize paragraph reaches a model whole --------------------


def _para(n_sentences: int, words_per: int = 12) -> str:
    return " ".join(
        f"Sentence {i} " + "word " * (words_per - 2) + "end." for i in range(n_sentences)
    )


class TestOversizeParagraphs:
    def test_split_at_sentence_ends_into_pieces_that_fit(self) -> None:
        para = _para(100)  # ~1,200 words
        pieces = split_oversize(para, 300)
        assert len(pieces) >= 4
        assert all(len(p.split()) <= 300 for p in pieces)
        assert " ".join(pieces) == para  # nothing lost, nothing invented
        assert all(p.endswith("end.") for p in pieces)

    def test_a_paragraph_that_fits_is_untouched(self) -> None:
        para = _para(5)
        assert split_oversize(para, 300) == [para]

    def test_a_lone_sentence_longer_than_the_window_stays_whole(self) -> None:
        sentence = "word " * 500 + "end."
        assert split_oversize(sentence, 300) == [sentence]

    def test_no_chunk_exceeds_target_plus_one_paragraph_and_a_short_tail(self) -> None:
        small = _para(3)  # ~36 words, under MIN_TAIL_WORDS
        huge = _para(400)  # ~4,800 words: the reference-list shape D19 names
        text = "\n\n".join([small, huge, small, huge, small])
        chunks = chunk_document("doc", text, target_words=300)
        assert max(c.n_words for c in chunks) <= 300 + 2 * len(small.split())
        assert len(chunks) > 30

    def test_split_pieces_are_never_carried_as_overlap(self) -> None:
        # Carrying a piece would make every chunk of a long paragraph twice its size.
        huge = _para(400)
        chunks = chunk_document("doc", huge, target_words=300)
        assert max(c.n_words for c in chunks) <= 300
        bodies = [c.text for c in chunks]
        assert len(set(bodies)) == len(bodies)
        assert " ".join(bodies) == huge  # every word exactly once

    def test_an_abbreviation_is_not_a_sentence_end(self) -> None:
        # The lookahead admits a digit and "(" for bibliographies, which is exactly
        # what makes "et al. (2019)" and "Fig. 3" look like boundaries.
        para = (
            "We grew yeast. Smith et al. (2019) reported growth. See Fig. 3 for the curve. "
            "J. Smith agreed. Done."
        )
        assert _sentences(para) == [
            "We grew yeast.",
            "Smith et al. (2019) reported growth.",
            "See Fig. 3 for the curve.",
            "J. Smith agreed.",
            "Done.",
        ]

    def test_a_bibliography_still_splits_at_its_numbered_entries(self) -> None:
        para = " ".join(
            f"[{i}] Author A. Title of the paper. Journal {i}, 1-9, 20{i:02d}."
            for i in range(1, 40)
        )
        pieces = split_oversize(para, 60)
        assert len(pieces) > 3
        assert " ".join(pieces) == para

    def test_a_short_tail_joins_the_chunk_before_it(self) -> None:
        # The defect this rule closes: a document whose last window held "2." shipped
        # that as a chunk of its own — one extraction call and one embedding for one
        # character of text. Proven against the rule disabled, so the assertion cannot
        # pass on a no-op (D11).
        text = _para(120) + "\n\nData availability\n\n" + _para(24)
        without = chunk_document("doc", text, target_words=300, min_tail_words=0)
        assert min(c.n_words for c in without) < MIN_TAIL_WORDS  # the defect, reproduced
        chunks = chunk_document("doc", text, target_words=300)
        assert all(c.n_words >= MIN_TAIL_WORDS for c in chunks)
        assert chunks[-1].text.endswith(without[-1].text)  # the tail moved, nothing was cut

    def test_every_chunk_adds_text_of_its_own(self) -> None:
        # The other half of the same rule, and the shape the demo corpus actually hit: a
        # heading paragraph flushed between two near-target blocks made a chunk that was
        # 99% the previous chunk's overlap — a second extraction call for two new words.
        # Distinct wording per block: identical sentences across paragraphs would make a
        # fresh paragraph look carried, and the measure below reads paragraph identity.
        first = " ".join(f"Alpha {i} " + "word " * 10 + "end." for i in range(24))
        second = " ".join(f"Beta {i} " + "word " * 10 + "end." for i in range(25))
        text = "\n\n".join([first, "Data availability", second])
        for chunks, rule in (
            (chunk_document("doc", text, target_words=300, min_tail_words=0), "off"),
            (chunk_document("doc", text, target_words=300), "on"),
        ):
            fresh = []
            for before, after in pairwise(chunks):
                carried = set(before.text.split("\n\n"))
                fresh.append(
                    sum(len(p.split()) for p in after.text.split("\n\n") if p not in carried)
                )
            if rule == "off":
                assert min(fresh) < MIN_TAIL_WORDS  # reproduced with the rule disabled
            else:
                assert all(f >= MIN_TAIL_WORDS for f in fresh), fresh

    def test_a_document_shorter_than_the_minimum_is_still_one_chunk(self) -> None:
        chunks = chunk_document("doc", "Three words only.", target_words=300)
        assert len(chunks) == 1
        assert chunks[0].text == "Three words only."

    def test_the_merge_never_loses_or_duplicates_a_paragraph(self) -> None:
        paras = [_para(20), _para(20), "A short closing note.", _para(2)]
        chunks = chunk_document("doc", "\n\n".join(paras), target_words=300)
        joined = "\n\n".join(c.text for c in chunks)
        for para in paras:
            assert para in joined, f"lost: {para[:40]}"
        # the carried overlap is the only text allowed to appear twice
        assert joined.count("A short closing note.") <= 2

    def test_an_unsplittable_oversize_paragraph_is_never_carried(self) -> None:
        """`split_oversize` returns one piece both when a paragraph FITS and when it is
        too big to cut. Treating the second as whole made the largest paragraph in a
        document the one most likely to be re-sent as overlap."""
        giant = "word " * 1500  # no sentence end anywhere: uncuttable
        chunks = chunk_document("doc", giant + "\n\n" + _para(60), target_words=300)
        bodies = [c.text for c in chunks]
        assert sum(1 for b in bodies if giant.strip() in b) == 1, "the giant was re-sent"

    def test_a_sentence_ending_in_one_character_is_joined_to_the_next(self) -> None:
        """A known, deliberate cost of the initials rule, pinned so it stays visible.

        "conducted in R." and "with pestle A." are real sentence ends that the rule
        rejects, because telling them from "J. Smith" needs to know a surname follows.
        It moves only where an oversize paragraph is cut — never what a chunk contains —
        so it is documented in `_sentences` rather than fixed with a bigger machine.
        """
        joined = _sentences("Analysis was conducted in R. The KEGG database was used.")
        assert len(joined) == 1  # the cost
        assert _sentences("Cells were washed. The pellet was kept.") == [
            "Cells were washed.",
            "The pellet was kept.",
        ]  # and the ordinary case still splits

    def test_whole_paragraphs_still_overlap(self) -> None:
        a, b, c = _para(20), _para(20), _para(20)  # ~240 words each
        chunks = chunk_document("doc", "\n\n".join([a, b, c]), target_words=300)
        assert chunks[1].text.startswith(a) or chunks[1].text.startswith(b)
        assert any(b in ch.text for ch in chunks[1:])


# --- one cache identity per wire, and the proof on every record ------------------


class TestExtractionDispatchOrder:
    """Longest-first dispatch must not change what the caller sees: the graph builder's
    relation pass only bumps an edge when both entities already exist, so a different
    extraction ORDER is a different graph."""

    def test_results_come_back_in_the_caller_s_order(self) -> None:
        from peerpanel.graph.extract import extract_many
        from peerpanel.text.chunks import Chunk

        chunks = [Chunk(f"d:{i}:{i:08x}", "d", i, "word " * (10 * (i % 7 + 1))) for i in range(20)]
        seen: list[str] = []

        class _Stub:
            name = "stub"

            def chat(self, *, system: str, user: str, max_tokens: int, **_: object) -> Any:
                seen.append(user)
                return ChatResponse(
                    text='{"entities": [], "relations": []}',
                    model="stub",
                    prompt_tokens=1,
                    completion_tokens=1,
                    context=4096,
                )

        out = extract_many(chunks, _Stub(), concurrency=1)  # type: ignore[arg-type]
        assert [e.chunk_id for e in out] == [c.chunk_id for c in chunks]
        # and the dispatch really was longest-first, or the reload argument is fiction
        dispatched = [len(u) for u in seen]
        assert dispatched == sorted(dispatched, reverse=True), dispatched

    def test_a_repeated_chunk_id_is_refused_not_silently_merged(self) -> None:
        """Keying results by chunk id is what preserves the caller's order; it is also how
        a duplicate id would disappear, since the returned list is still the right LENGTH
        with one extraction standing in for two."""
        from peerpanel.graph.extract import extract_many
        from peerpanel.text.chunks import Chunk

        twins = [
            Chunk("d:0:aaaaaaaa", "d", 0, "first text"),
            Chunk("d:0:aaaaaaaa", "d", 0, "other"),
        ]

        class _Stub:
            name = "stub"

            def chat(self, *, system: str, user: str, max_tokens: int, **_: object) -> Any:
                return ChatResponse(
                    text='{"entities": [], "relations": []}',
                    model="stub",
                    prompt_tokens=1,
                    completion_tokens=1,
                    context=4096,
                )

        with pytest.raises(ValueError, match="chunk id repeats"):
            extract_many(twins, _Stub(), concurrency=1)  # type: ignore[arg-type]


class TestCacheIdentities:
    def test_extraction_records_are_filed_under_the_native_wire(self) -> None:
        assert OllamaNativeChat(EXTRACTION_MODEL).name == EXTRACTION_PROVIDER_NAME
        assert EXTRACTION_PROVIDER_NAME.startswith("ollama-native:")

    def test_community_reports_are_filed_under_the_openai_wire(self) -> None:
        assert OllamaOpenAIChat(SUMMARY_MODEL).name == SUMMARY_PROVIDER_NAME
        assert SUMMARY_PROVIDER_NAME.startswith("ollama-openai:")
        assert EXTRACTION_MODEL == SUMMARY_MODEL  # one model, two wires (D3)


class TestCallStatsOnEveryRecord:
    def test_from_ledger_carries_the_proof(self) -> None:
        ledger = TokenLedger()
        ledger.record(
            ChatResponse(text="", model="m", prompt_tokens=4_100, completion_tokens=1, context=8192)
        )
        ledger.record(
            ChatResponse(text="", model="m", prompt_tokens=900, completion_tokens=1, context=4096)
        )
        stats = CallStats.from_ledger(ledger)
        # The margin is per call (4096-900=3196), NOT largest-prompt vs smallest-window,
        # which would read -4 here and call two honest calls a cut prompt.
        assert stats == CallStats(
            calls=2,
            largest_prompt_tokens=4_100,
            smallest_context=4096,
            smallest_headroom_tokens=3196,
        )
        assert stats.largest_prompt_tokens > (stats.smallest_context or 0)

    def test_the_margin_goes_negative_only_when_a_prompt_did_not_fit(self) -> None:
        ledger = TokenLedger()
        ledger.record(
            ChatResponse(text="", model="m", prompt_tokens=4_100, completion_tokens=1, context=4096)
        )
        assert CallStats.from_ledger(ledger).smallest_headroom_tokens == -4

    def test_a_panel_record_without_it_cannot_load(self) -> None:
        fields = {
            "manuscript_doi": "10.1/x",
            "corpus_manifest": "sha",
            "excluded_docs": [],
            "reviewer_outputs": [],
            "verdicts": [],
            "deterministic_findings": [],
            "conflicts": [],
            "summary": "",
            "swap_consistency_rate": None,
            "n_swap_checked": 0,
            "total_tokens": 0,
            "wall_s": 0.0,
        }
        with pytest.raises(ValueError):
            PanelReview.model_validate(fields)
        PanelReview.model_validate(
            {
                **fields,
                "model_calls": {
                    "calls": 0,
                    "largest_prompt_tokens": 0,
                    "smallest_context": None,
                    "smallest_headroom_tokens": None,
                },
            }
        )
