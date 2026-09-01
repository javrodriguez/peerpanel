"""The planted-error benchmark must be able to earn the score it reports.

Three ways a planted-error harness can report a number that means nothing, all
of which this one did at some point:

1. Plant errors where no reviewer looks. Both arms read a 900-word excerpt;
   planting scanned the whole 8,000-word body, so every error landed outside the
   window and neither arm could possibly have found one.
2. Score channels that are not assertions. A verdict quoting the manuscript is a
   restatement, so a SUPPORTS verdict — the panel AGREEING with a planted error —
   counted as a catch, on a channel the baseline does not even have.
3. Use a detection token common enough to appear by accident. "decreased" occurs
   in 37 of the 68 corpus documents, so either arm quoting retrieved literature
   could mint a catch nobody earned.

These tests pin all three as properties of the harness, so a future change that
reintroduces any of them fails here rather than silently producing a number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from peerpanel.agents.reviewer_base import EXCERPT_WORDS, excerpt
from peerpanel.evals.planted import detect, plant_errors
from peerpanel.manuscripts.store import read_manuscript

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = ROOT / "manuscripts" / "caprin-heterochromatin.txt"


def _planted():  # type: ignore[no-untyped-def]
    _header, body = read_manuscript(SUBJECT)
    return plant_errors(body, SUBJECT.name, window_words=EXCERPT_WORDS)


class TestErrorsAreReachable:
    def test_every_planted_error_is_inside_the_reviewed_window(self) -> None:
        planted = _planted()
        assert planted.errors, "nothing planted — the benchmark would score 0/0"
        visible = excerpt(planted.text)
        unreachable = [e.error_id for e in planted.errors if e.detection_token not in visible]
        assert not unreachable, (
            f"{unreachable} lie outside the {EXCERPT_WORDS}-word window both arms read"
        )

    def test_kinds_that_cannot_be_planted_are_recorded_not_dropped(self) -> None:
        """A silently absent error kind makes the denominator lie."""
        planted = _planted()
        kinds = {e.kind for e in planted.errors}
        all_kinds = kinds | set(planted.skipped_kinds)
        assert len(all_kinds) == 5, all_kinds
        for kind, reason in planted.skipped_kinds.items():
            assert reason, f"{kind} skipped with no reason given"


class TestTokensCannotBeMintedByAccident:
    def test_no_detection_token_appears_in_the_corpus(self) -> None:
        """A token that occurs in retrieved literature can be quoted into a
        finding by either arm, crediting a catch nobody made."""
        planted = _planted()
        docs = sorted((ROOT / "corpus" / "demo").glob("*.txt"))
        if not docs:
            pytest.skip("demo corpus not fetched (see `make corpus`)")
        for error in planted.errors:
            hits = [
                d.name
                for d in docs
                if error.detection_token.lower() in d.read_text(errors="ignore").lower()
            ]
            assert not hits, (
                f"{error.kind}'s token {error.detection_token!r} occurs in "
                f"{len(hits)} corpus documents — it could be quoted, not found"
            )

    def test_a_direction_token_is_a_phrase_not_a_bare_word(self) -> None:
        planted = _planted()
        for error in planted.errors:
            if error.kind == "effect_direction_flip":
                assert len(error.detection_token.split()) > 1, (
                    "a bare direction word is common enough to appear by accident"
                )


class TestScoringCreditsOnlyAssertions:
    def test_restating_the_error_is_not_a_catch(self) -> None:
        """The channel rule, at the level of `detect` itself: a catch requires the
        token to appear in text the ARM ASSERTED, and the harness only ever passes
        assertion channels. Here we prove the negative case is representable."""
        planted = _planted()
        error = planted.errors[0]
        restatement = [f"SUPPORTS {error.detection_token} is stated in the manuscript"]
        # detect() is a substring rule by design, so the protection cannot live
        # here — it lives in WHICH texts the harness collects. This test documents
        # that boundary so a future change cannot quietly widen the channels.
        assert detect(error, restatement), "detect is a substring rule"
        from peerpanel.evals import planted_eval

        source = Path(planted_eval.__file__).read_text()
        assert 'v.verdict == "REFUTES"' in source, (
            "the panel's scored channels must exclude non-REFUTES verdicts"
        )
        assert "review.summary" not in source.split("panel_texts")[1][:400], (
            "the converger's prose is not an assertion channel"
        )
