"""Claims-verifier tests — the swap law, span grounding and honest failure are
proven against a unit stub of our own chat seam; the live road is ollama-marked."""

from __future__ import annotations

import hashlib
import json

import pytest

from peerpanel.agents.claims_verifier import (
    MIN_SWAP_N,
    REVIEWER_CLAIM_RULE,
    VERDICT_SCHEMA,
    decompose_claims,
    decomposition_schema,
    swap_consistency_by_source,
    swap_consistency_rate,
    verify_claim,
    verify_claims,
)
from peerpanel.agents.schemas import (
    VERDICT_NEI,
    VERDICT_REFUTES,
    VERDICT_SUPPORTS,
    AtomicClaim,
    ClaimVerdict,
)
from peerpanel.providers.base import ChatResponse, TokenLedger


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
            {
                "system": system,
                "user": user,
                "schema": json_schema,
                "temp": temperature,
                "max_tokens": max_tokens,
            }
        )
        payload = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        return ChatResponse(text=payload, model="stub", prompt_tokens=3, completion_tokens=5)


PASSAGE = (
    "Met4 activates the sulfur metabolism regulon. Deletion of MET17 abolishes "
    "growth on minimal medium."
)

C1_ID = "d1:0:aaaa0001"
C1_TEXT = "Met4 activates the sulfur metabolism regulon in Saccharomyces cerevisiae."
C2_ID = "d2:0:bbbb0002"
C2_TEXT = "Cbf1 binds the MET17 promoter together with Met31 under sulfur limitation."
HITS = [(C1_ID, C1_TEXT), (C2_ID, C2_TEXT)]

CLAIM = AtomicClaim(
    claim_id="manuscript:0:12345678",
    text="Met4 activates the sulfur metabolism regulon.",
    source="manuscript",
)

TRUNCATED = '{"claims": ["Met4 activates the sulf'


def _claims_payload(*claims: str) -> str:
    return json.dumps({"claims": list(claims)})


def _verdict_payload(verdict: str, *spans: tuple[str, str]) -> str:
    return json.dumps(
        {"verdict": verdict, "spans": [{"chunk_id": c, "quote": q} for c, q in spans]}
    )


SUPPORTS_C1 = _verdict_payload(VERDICT_SUPPORTS, (C1_ID, "Met4 activates the sulfur metabolism"))
REFUTES_C2 = _verdict_payload(VERDICT_REFUTES, (C2_ID, "Cbf1 binds the MET17 promoter"))


class TestDecomposition:
    def test_schema_enforces_the_cap_at_temperature_zero(self) -> None:
        stub = _StubChat(_claims_payload("a claim", "another claim"))
        decompose_claims(PASSAGE, stub, "manuscript", max_claims=3)
        call = stub.calls[0]
        assert call["schema"] == decomposition_schema(3)
        assert decomposition_schema(3)["properties"]["claims"]["maxItems"] == 3
        assert call["temp"] == 0.0
        assert "AT MOST 3" in str(call["system"])
        assert PASSAGE in str(call["user"])

    def test_claim_id_carries_source_index_and_passage_hash(self) -> None:
        stub = _StubChat(_claims_payload("first claim", "second claim"))
        claims = decompose_claims(PASSAGE, stub, "manuscript")
        expected = hashlib.sha256(PASSAGE.encode()).hexdigest()[:8]
        assert [c.claim_id for c in claims] == [
            f"manuscript:0:{expected}",
            f"manuscript:1:{expected}",
        ]
        assert [c.text for c in claims] == ["first claim", "second claim"]
        assert {c.source for c in claims} == {"manuscript"}

    def test_source_names_the_producer(self) -> None:
        stub = _StubChat(_claims_payload("a reviewer assertion"))
        claims = decompose_claims(PASSAGE, stub, "methods-statistics")
        assert claims[0].claim_id.startswith("methods-statistics:0:")
        assert claims[0].source == "methods-statistics"

    def test_overlong_payload_is_cut_to_the_cap(self) -> None:
        stub = _StubChat(_claims_payload(*(f"claim {i}" for i in range(9))))
        claims = decompose_claims(PASSAGE, stub, "manuscript", max_claims=2)
        assert [c.text for c in claims] == ["claim 0", "claim 1"]

    def test_blank_and_non_string_entries_dropped(self) -> None:
        payload = json.dumps({"claims": ["  ", 7, None, "  a real claim  "]})
        claims = decompose_claims(PASSAGE, _StubChat(payload), "manuscript")
        assert [c.text for c in claims] == ["a real claim"]

    def test_duplicate_claims_collapse(self) -> None:
        stub = _StubChat(_claims_payload("Met4 activates", "met4 ACTIVATES", "a second"))
        claims = decompose_claims(PASSAGE, stub, "manuscript")
        assert [c.text for c in claims] == ["Met4 activates", "a second"]
        assert [c.claim_id.split(":")[1] for c in claims] == ["0", "1"]

    def test_truncated_json_retries_at_double_budget_then_succeeds(self) -> None:
        stub = _StubChat(TRUNCATED, _claims_payload("a recovered claim"))
        claims = decompose_claims(PASSAGE, stub, "manuscript")
        assert len(stub.calls) == 2
        assert stub.calls[0]["max_tokens"] == 1024
        assert stub.calls[1]["max_tokens"] == 2048
        assert [c.text for c in claims] == ["a recovered claim"]

    def test_double_failure_is_no_claims_not_a_crash(self) -> None:
        stub = _StubChat(TRUNCATED)
        assert decompose_claims(PASSAGE, stub, "manuscript") == []
        assert len(stub.calls) == 2

    def test_non_dict_and_non_list_payloads_yield_nothing(self) -> None:
        assert decompose_claims(PASSAGE, _StubChat('["a list"]'), "manuscript") == []
        assert decompose_claims(PASSAGE, _StubChat('{"claims": "a string"}'), "manuscript") == []


class TestSwapAgreement:
    def test_both_orders_agree_so_the_verdict_stands(self) -> None:
        stub = _StubChat(SUPPORTS_C1)
        out = verify_claim(CLAIM, HITS, stub, TokenLedger())
        assert len(stub.calls) == 2
        assert out.verdict == VERDICT_SUPPORTS
        assert out.swap_consistent is True
        assert [s.chunk_id for s in out.evidence] == [C1_ID]
        assert out.claim_id == CLAIM.claim_id
        assert out.claim_text == CLAIM.text

    def test_second_call_reverses_the_evidence_order(self) -> None:
        stub = _StubChat(SUPPORTS_C1)
        verify_claim(CLAIM, HITS, stub, None)
        first, second = str(stub.calls[0]["user"]), str(stub.calls[1]["user"])
        assert first.index(f"[{C1_ID}]") < first.index(f"[{C2_ID}]")
        assert second.index(f"[{C2_ID}]") < second.index(f"[{C1_ID}]")
        assert C1_TEXT in first and C1_TEXT in second

    def test_both_calls_carry_the_verdict_schema_at_temperature_zero(self) -> None:
        stub = _StubChat(SUPPORTS_C1)
        verify_claim(CLAIM, HITS, stub, None)
        assert all(c["schema"] == VERDICT_SCHEMA for c in stub.calls)
        assert all(c["temp"] == 0.0 for c in stub.calls)
        assert VERDICT_SCHEMA["properties"]["verdict"]["enum"] == [
            VERDICT_SUPPORTS,
            VERDICT_REFUTES,
            VERDICT_NEI,
        ]

    def test_output_spans_follow_the_given_order_not_the_call_order(self) -> None:
        # forward cites the second chunk, reverse cites the first: the output is
        # sorted by the order the retriever gave, so it cannot leak call order.
        forward = _verdict_payload(VERDICT_SUPPORTS, (C2_ID, "Cbf1 binds the MET17 promoter"))
        reverse = _verdict_payload(VERDICT_SUPPORTS, (C1_ID, "Met4 activates"))
        out = verify_claim(CLAIM, HITS, _StubChat(forward, reverse), None)
        assert [s.chunk_id for s in out.evidence] == [C1_ID, C2_ID]
        assert out.swap_consistent is True

    def test_identical_spans_from_both_orders_appear_once(self) -> None:
        out = verify_claim(CLAIM, HITS, _StubChat(SUPPORTS_C1), None)
        assert len(out.evidence) == 1

    def test_no_evidence_abstains_without_spending_a_call(self) -> None:
        stub = _StubChat(SUPPORTS_C1)
        out = verify_claim(CLAIM, [], stub, TokenLedger())
        assert stub.calls == []
        assert out.verdict == VERDICT_NEI
        assert out.evidence == []
        assert out.swap_consistent is True  # no order existed to disagree about


class TestSwapDisagreement:
    def test_flipping_the_order_flips_the_verdict_so_the_claim_abstains(self) -> None:
        stub = _StubChat(SUPPORTS_C1, REFUTES_C2)
        out = verify_claim(CLAIM, HITS, stub, None)
        assert len(stub.calls) == 2
        assert out.verdict == VERDICT_NEI
        assert out.swap_consistent is False

    def test_abstain_keeps_only_spans_both_orders_cited(self) -> None:
        forward = _verdict_payload(
            VERDICT_SUPPORTS,
            (C1_ID, "Met4 activates"),
            (C2_ID, "Cbf1 binds the MET17 promoter"),
        )
        reverse = _verdict_payload(VERDICT_REFUTES, (C2_ID, "Met31 under sulfur limitation"))
        out = verify_claim(CLAIM, HITS, _StubChat(forward, reverse), None)
        assert out.verdict == VERDICT_NEI
        assert out.swap_consistent is False
        assert {s.chunk_id for s in out.evidence} == {C2_ID}
        assert len(out.evidence) == 2  # both order-invariant quotes on that chunk

    def test_a_side_that_never_parses_can_only_abstain(self) -> None:
        stub = _StubChat(SUPPORTS_C1, '{"verdict": "SUPP')
        out = verify_claim(CLAIM, HITS, stub, None)
        assert len(stub.calls) == 3  # forward once, reverse twice (the doubled retry)
        assert out.verdict == VERDICT_NEI
        assert out.swap_consistent is False
        assert out.evidence == []

    def test_an_out_of_vocabulary_verdict_abstains(self) -> None:
        stub = _StubChat(SUPPORTS_C1, _verdict_payload("MAYBE", (C1_ID, "Met4 activates")))
        out = verify_claim(CLAIM, HITS, stub, None)
        assert out.verdict == VERDICT_NEI
        assert out.swap_consistent is False

    def test_both_sides_unparseable_agree_on_abstaining(self) -> None:
        stub = _StubChat("not json at all")
        out = verify_claim(CLAIM, HITS, stub, None)
        assert len(stub.calls) == 4  # two judgements, each with its one retry
        assert out.verdict == VERDICT_NEI
        assert out.swap_consistent is True


class TestSpanGrounding:
    def test_hallucinated_chunk_id_is_filtered(self) -> None:
        payload = _verdict_payload(
            VERDICT_SUPPORTS,
            ("d9:0:cccc0009", "a chunk that was never given"),
            (C1_ID, "Met4 activates"),
        )
        out = verify_claim(CLAIM, HITS, _StubChat(payload), None)
        assert [s.chunk_id for s in out.evidence] == [C1_ID]

    def test_a_verdict_left_with_no_grounded_span_is_returned_visibly_ungrounded(self) -> None:
        payload = _verdict_payload(VERDICT_SUPPORTS, ("d9:0:cccc0009", "invented"))
        out = verify_claim(CLAIM, HITS, _StubChat(payload), None)
        assert out.verdict == VERDICT_SUPPORTS
        assert out.evidence == []

    def test_non_substring_quote_is_dropped(self) -> None:
        payload = _verdict_payload(
            VERDICT_SUPPORTS,
            (C1_ID, "Met4 represses sulfur metabolism"),  # paraphrase-as-quote
            (C1_ID, "sulfur metabolism regulon"),
        )
        out = verify_claim(CLAIM, HITS, _StubChat(payload), None)
        assert [s.quote for s in out.evidence] == ["sulfur metabolism regulon"]

    def test_quote_from_the_wrong_chunk_is_dropped(self) -> None:
        payload = _verdict_payload(VERDICT_SUPPORTS, (C2_ID, "Met4 activates"))
        out = verify_claim(CLAIM, HITS, _StubChat(payload), None)
        assert out.evidence == []

    def test_rewrapped_quote_survives_whitespace_normalisation(self) -> None:
        payload = _verdict_payload(VERDICT_SUPPORTS, (C1_ID, "Met4   activates\nthe sulfur"))
        out = verify_claim(CLAIM, HITS, _StubChat(payload), None)
        assert len(out.evidence) == 1

    def test_empty_quote_and_malformed_span_dropped(self) -> None:
        payload = json.dumps(
            {"verdict": VERDICT_SUPPORTS, "spans": [{"chunk_id": C1_ID, "quote": " "}, "junk"]}
        )
        out = verify_claim(CLAIM, HITS, _StubChat(payload), None)
        assert out.evidence == []


class TestLedger:
    def test_every_call_is_recorded(self) -> None:
        ledger = TokenLedger()
        decompose_claims(
            PASSAGE, _StubChat(TRUNCATED, _claims_payload("c")), "manuscript", ledger=ledger
        )
        verify_claim(CLAIM, HITS, _StubChat(SUPPORTS_C1), ledger)
        assert ledger.calls == 4  # 2 decomposition (retry) + 2 judgements
        assert ledger.prompt_tokens == 12
        assert ledger.completion_tokens == 20
        assert ledger.total_tokens == 32

    def test_ledger_is_optional(self) -> None:
        out = verify_claim(CLAIM, HITS, _StubChat(SUPPORTS_C1), None)
        assert out.verdict == VERDICT_SUPPORTS


class TestVerifyClaims:
    def test_each_claim_is_retrieved_for_by_its_own_text(self) -> None:
        queries: list[str] = []

        def retrieve(query: str) -> list[tuple[str, str]]:
            queries.append(query)
            return HITS

        claims = [
            CLAIM,
            AtomicClaim(claim_id="manuscript:1:12345678", text="MET17 deletion", source="m"),
        ]
        stub = _StubChat(SUPPORTS_C1)
        verdicts = verify_claims(claims, retrieve, stub, TokenLedger())
        assert queries == [c.text for c in claims]
        assert [v.claim_id for v in verdicts] == [c.claim_id for c in claims]
        assert len(stub.calls) == 4  # two claims, judged twice each

    def test_a_claim_with_no_retrieved_evidence_abstains(self) -> None:
        stub = _StubChat(SUPPORTS_C1)
        verdicts = verify_claims([CLAIM], lambda _q: [], stub, None)
        assert stub.calls == []
        assert verdicts[0].verdict == VERDICT_NEI
        assert verdicts[0].swap_consistent is True
        assert verdicts[0].swapped is False

    def test_concurrent_verification_keeps_input_order_and_exact_counts(self) -> None:
        """Claims are judged VERIFY_WORKERS at a time; the record must read in
        input order whichever finished first, and the shared ledger must count
        every call exactly once under contention."""
        import threading
        import time

        class _SlowStub(_StubChat):
            def chat(self, **kw: object) -> ChatResponse:  # type: ignore[override]
                # Later claims answer sooner, so completion order is reversed.
                claim = str(kw["user"]).split("\n")[1]
                time.sleep(0.02 * (12 - int(claim.rsplit("#", 1)[1])))
                return super().chat(**kw)  # type: ignore[arg-type]

        claims = [
            AtomicClaim(claim_id=f"manuscript:{i}:12345678", text=f"claim #{i}", source="m")
            for i in range(12)
        ]
        stub = _SlowStub(SUPPORTS_C1)
        ledger = TokenLedger()
        seen_threads: set[int] = set()

        def retrieve(_q: str) -> list[tuple[str, str]]:
            seen_threads.add(threading.get_ident())
            return HITS

        verdicts = verify_claims(claims, retrieve, stub, ledger)
        assert [v.claim_id for v in verdicts] == [c.claim_id for c in claims]
        assert all(v.claim_text == c.text for v, c in zip(verdicts, claims, strict=True))
        assert len(seen_threads) > 1, "verification did not actually run concurrently"
        assert ledger.calls == 24 and ledger.total_tokens == 24 * 8


def _verdict(consistent: bool, source: str = "m") -> ClaimVerdict:
    return ClaimVerdict(
        claim_id=f"{source}:0:12345678",
        claim_text="a claim",
        verdict=VERDICT_SUPPORTS if consistent else VERDICT_NEI,
        evidence=[],
        swap_consistent=consistent,
        swapped=True,
    )


class TestSwapConsistencyBySource:
    def test_each_source_gets_its_own_rate_and_n(self) -> None:
        verdicts = (
            [_verdict(True, "manuscript")] * MIN_SWAP_N
            + [_verdict(False, "methods-statistics")] * MIN_SWAP_N
            + [_verdict(True, "prior-work-novelty")] * 2
        )
        by = swap_consistency_by_source(verdicts)
        assert by["manuscript"] == (1.0, MIN_SWAP_N)
        assert by["methods-statistics"] == (0.0, MIN_SWAP_N)
        assert by["prior-work-novelty"] == (None, 2)  # below the reporting n: withheld

    def test_unswapped_claims_do_not_count_toward_any_source(self) -> None:
        unswapped = ClaimVerdict(
            claim_id="manuscript:0:12345678",
            claim_text="c",
            verdict=VERDICT_NEI,
            evidence=[],
            swap_consistent=True,
            swapped=False,
        )
        assert swap_consistency_by_source([unswapped]) == {"manuscript": (None, 0)}

    def test_reviewer_passages_get_the_science_only_rule(self) -> None:
        """Reviewer findings are opinions; only the propositions about the science
        inside them are claims. The rule is in the prompt for every non-manuscript
        source and absent for the manuscript's own text."""
        stub = _StubChat(_claims_payload("Dcr1 is required for heterochromatin."))
        decompose_claims("The intro is well organised.", stub, source="methods-statistics")
        decompose_claims(PASSAGE, stub, source="manuscript")
        assert REVIEWER_CLAIM_RULE.strip() in str(stub.calls[0]["system"])
        assert "reviewer's notes" not in str(stub.calls[1]["system"])


class TestSwapConsistencyRate:
    def test_rate_withheld_below_the_reporting_n(self) -> None:
        rate, n = swap_consistency_rate([_verdict(True)] * (MIN_SWAP_N - 1))
        assert rate is None
        assert n == MIN_SWAP_N - 1

    def test_empty_input_reports_zero_and_no_rate(self) -> None:
        assert swap_consistency_rate([]) == (None, 0)

    def test_rate_reported_at_the_threshold(self) -> None:
        verdicts = [_verdict(True)] * 8 + [_verdict(False)] * 2
        rate, n = swap_consistency_rate(verdicts)
        assert n == MIN_SWAP_N == 10
        assert rate == pytest.approx(0.8)

    def test_all_inconsistent_is_zero_not_none(self) -> None:
        rate, n = swap_consistency_rate([_verdict(False)] * 12)
        assert rate == pytest.approx(0.0)
        assert n == 12


@pytest.mark.ollama
class TestLiveVerification:
    """Real llama3.1:8b, one claim its evidence plainly supports.

    Regenerate: `uv run pytest tests/test_claims_verifier.py -m ollama` with
    `ollama pull llama3.1:8b` done and the server up.
    """

    def test_obvious_support_survives_the_swap(self) -> None:
        from peerpanel.providers import OllamaOpenAIChat

        ledger = TokenLedger()
        claim = AtomicClaim(
            claim_id="live:0:00000000",
            text="Met4 activates sulfur metabolism in Saccharomyces cerevisiae.",
            source="manuscript",
        )
        hits = [
            ("live:0", C1_TEXT),
            ("live:1", "Ribosome biogenesis is coordinated with nutrient availability."),
        ]
        out = verify_claim(claim, hits, OllamaOpenAIChat("llama3.1:8b"), ledger)
        assert out.verdict == VERDICT_SUPPORTS
        assert out.swap_consistent is True
        assert [s.chunk_id for s in out.evidence] == ["live:0"]
        assert ledger.calls == 2

    def test_real_decomposition_returns_atomic_claims(self) -> None:
        from peerpanel.providers import OllamaOpenAIChat

        claims = decompose_claims(
            PASSAGE, OllamaOpenAIChat("llama3.1:8b"), "manuscript", max_claims=4
        )
        assert 1 <= len(claims) <= 4
        assert all(c.claim_id.startswith("manuscript:") for c in claims)
        assert any("met4" in c.text.lower() or "met17" in c.text.lower() for c in claims)
