"""The planted-error benchmark must be able to earn the score it reports.

Four ways a planted-error harness can report a number that means nothing, all
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
4. Credit a restatement as a detection. The substring rule (`detect`) cannot tell
   "the excerpt mentions Dcr2" from "Dcr2 is the wrong symbol", and round 3 found
   that every credited catch in the first committed records was of the first kind
   — a verbatim quotation of the perturbed manuscript.

These tests pin all four as properties of the harness, so a future change that
reintroduces any of them fails here rather than silently producing a number.

The fourth is the reason for `asserts` and for the two controls it is held to:

- the NEGATIVE control, `tests/fixtures/round3_scored_strings.json` — every one of
  the 230 strings the round-3 records scored, at the committed bytes, none of which
  may be credited by the assertion rule. It is a FLOOR, not evidence: not one of
  those strings contains an assertion cue at all, which the test below measures
  rather than assumes;
- the BOUNDARY and POSITIVE controls, which therefore carry the weight. They are
  built from REAL text — sentences pulled out of the committed manuscripts by the
  phrase they collide with, and real record strings with a genuine assertion
  sentence appended — never from prose invented to make a rule look good.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from peerpanel.agents.reviewer_base import EXCERPT_WORDS, excerpt
from peerpanel.evals.collisions import token_collisions
from peerpanel.evals.planted import (
    ASSERTION_CUES,
    DETECTION_RULE,
    PlantedError,
    asserts,
    detect,
    plant_errors,
)
from peerpanel.manuscripts.store import read_manuscript
from peerpanel.text.chunks import sentences

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPTS = ROOT / "manuscripts"
SUBJECT = MANUSCRIPTS / "caprin-heterochromatin.txt"
SUBJECT_STEMS = sorted(path.stem for path in MANUSCRIPTS.glob("*.txt"))
FIXTURE = ROOT / "tests" / "fixtures" / "round3_scored_strings.json"
COLLISIONS = ROOT / "corpus" / "token_collisions.json"
# The commit the round-3 planted-eval records were published at; the fixture is those
# records' bytes, and the test below re-derives it from git rather than trusting it.
ROUND_3_COMMIT = "52238c8bb6a5e952a6398ed7be1eed27d021202e"
ROUND_3_RECORDS = (
    "results/planted-eval-caprin-heterochromatin.json",
    "results/planted-eval-met17-auxotroph.json",
)
# The frozen alternation, written out here so a change to the cue list has to change a
# test that says it is frozen. D-1: the rule is specified once, frozen, and only then
# run against a control; a control that trips is a finding about the rule, never a
# licence to edit this text.
FROZEN_CUES = (
    r"\bincorrect\b|\bnot correct\b|\bwrong(ly)?\b|\berroneous(ly)?\b|"
    r"\berror\b(?!\s*bars?)|\bmistake[sn]?\b|\btypo\b|\bmisnam|\bmislabel|"
    r"\bmisidentif|\bshould (?:be|read)\b|\binconsisten|\bcontradict|\breversed\b|"
    r"\breversal\b|\bimplausib|\bimpossib|\bfabricat|\bdoes not exist\b|"
    r"\bnot (?:a )?(?:real|valid|known|traceable|verifiable)\b|"
    r"\bcannot (?:be )?(?:find|found|locate[d]?|verif(?:y|ied))\b|"
    r"\bcould not (?:be )?(?:find|found|locate[d]?|verif(?:y|ied))\b|"
    r"\buntraceab|\bunverifiab|\binvalid\b|\bmisnomer\b"
)
# Phrases that live in these manuscripts as ordinary vocabulary and that a careless cue
# list would fire on. The controls below are the real sentences carrying them.
COLLIDING_PHRASES = ("error bars", "reverse transcribed", "inverted microscope", "no such")


def _planted(path: Path = SUBJECT):  # type: ignore[no-untyped-def]
    _header, body = read_manuscript(path)
    return plant_errors(body, path.name, window_words=EXCERPT_WORDS)


def _fixture() -> dict[str, Any]:
    if not FIXTURE.exists():  # pragma: no cover - the fixture is committed
        pytest.fail(f"{FIXTURE} is missing — the negative control cannot run without it")
    return json.loads(FIXTURE.read_text())


def _errors_of(fixture: dict[str, Any], record: str) -> list[PlantedError]:
    """The planted errors each string was scored against, from the fixture's own tokens."""
    return [PlantedError(**error) for error in fixture["errors"][record]]


def _git_show(path: str) -> str | None:
    """The committed bytes of `path` at the round-3 commit, or None if unreachable."""
    try:
        done = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{ROUND_3_COMMIT}:{path}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - no git on the machine
        return None
    return done.stdout if done.returncode == 0 else None


def _real_manuscript_sentence(phrase: str) -> tuple[str, str]:
    """A committed manuscript sentence containing `phrase`, with its file and line.

    Grep, not invention: the control is only a control if the colliding phrase is text
    this repository actually ships. The citation is returned so a failure names the line
    a reader can open.
    """
    for path in sorted(MANUSCRIPTS.glob("*.txt")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if phrase.lower() not in line.lower():
                continue
            for sentence in sentences(line):
                if phrase.lower() in sentence.lower():
                    return f"{path.name}:{line_no}", sentence.strip()
    pytest.fail(f"no committed manuscript contains {phrase!r} — this control would be invented")


def _beside_the_token(sentence: str, token: str) -> str:
    """The real sentence with a planted token named inside it — ONE sentence, on purpose.

    This is the shape that must score 0: an arm naming the token in the same sentence as
    a phrase that collides with a dropped cue.
    """
    return f"{sentence.rstrip().removesuffix('.')}, as reported for {token}."


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
    def test_no_planted_token_occurs_in_the_original_manuscript(self) -> None:
        """A token already in the unperturbed paper is quotable, not findable.

        Unconditional: every manuscript this repository ships is planted and checked,
        with no corpus fetch and no network involved.
        """
        assert SUBJECT_STEMS, "no manuscripts committed — this guard would check nothing"
        checked = 0
        for stem in SUBJECT_STEMS:
            path = MANUSCRIPTS / f"{stem}.txt"
            _header, body = read_manuscript(path)
            planted = plant_errors(body, path.name, window_words=EXCERPT_WORDS)
            assert planted.errors, f"{stem}: nothing planted"
            for error in planted.errors:
                checked += 1
                assert error.detection_token.lower() not in body.lower(), (
                    f"{stem}: {error.kind}'s token {error.detection_token!r} is already in "
                    "the ORIGINAL manuscript — quoting the paper would mint a catch"
                )
        assert checked >= 5, f"only {checked} tokens checked"

    def test_no_planted_token_occurs_in_the_committed_ci_corpus(self) -> None:
        """The tracked corpus is checked live, in every clone, with no fetch."""
        report = token_collisions(ROOT, "ci", SUBJECT_STEMS)
        assert report["documents_scanned"] > 0, "the CI corpus scan read nothing"
        assert set(report["subjects"]) == set(SUBJECT_STEMS)
        for stem, tokens in report["subjects"].items():
            assert tokens, f"{stem}: no tokens scanned"
            for token, docs in tokens.items():
                assert not docs, f"{stem}: {token!r} occurs in {docs} — it could be quoted"

    def test_no_planted_token_occurs_in_the_corpus(self) -> None:
        """The fetched corpus, read from the committed measurement — never skipped.

        This guard used to skip whenever `corpus/demo` was absent, which is every clean
        clone and every CI run: the check the evaluation leans on never ran (round-3
        F12). The measurement is now a committed, model-free artifact, so the guard runs
        unconditionally and re-derives it whenever the corpus is present.
        """
        if not COLLISIONS.exists():
            pytest.fail(
                f"{COLLISIONS.relative_to(ROOT)} is missing — `make token-collisions` writes "
                "it (CP2 T2.0). Without it this guard would silently skip, which is the "
                "defect it exists to close."
            )
        report = json.loads(COLLISIONS.read_text())
        assert report["corpus"] == "demo", report["corpus"]
        assert report["documents_scanned"] > 0, "the committed scan read no documents"
        assert set(report["subjects"]) == set(SUBJECT_STEMS), (
            "the committed collision report does not cover every manuscript"
        )
        for stem, tokens in report["subjects"].items():
            assert tokens, f"{stem}: no tokens scanned"
            for token, docs in tokens.items():
                assert not docs, f"{stem}: {token!r} occurs in {docs} — it could be quoted"
        if sorted((ROOT / "corpus" / "demo").glob("*.txt")):
            assert token_collisions(ROOT, "demo", SUBJECT_STEMS) == report, (
                "corpus/token_collisions.json is stale — re-run `make token-collisions`"
            )

    def test_the_scan_reports_a_collision_when_there_is_one(self, tmp_path: Path) -> None:
        """Every guard above asserts an EMPTY list — so prove the scan can be non-empty.

        A collision report that always came back clean would pass every test in this
        class while measuring nothing, which is the defect class round 3 named.
        """
        shutil.copytree(MANUSCRIPTS, tmp_path / "manuscripts")
        corpus = tmp_path / "corpus" / "tiny"
        corpus.mkdir(parents=True)
        planted = _planted()
        token = next(e.detection_token for e in planted.errors if e.kind == "gene_symbol_swap")
        (corpus / "hit.txt").write_text(f"Processed by Dicer ({token.lower()}) in fission yeast.\n")
        (corpus / "miss.txt").write_text("Nothing in this document names any planted token.\n")
        report = token_collisions(tmp_path, "tiny", ["caprin-heterochromatin"])
        assert report["documents_scanned"] == 2
        assert report["subjects"]["caprin-heterochromatin"][token] == ["hit.txt"]
        other = [
            docs
            for tok, docs in report["subjects"]["caprin-heterochromatin"].items()
            if tok != token
        ]
        assert other and all(not docs for docs in other), other

    def test_an_empty_corpus_raises_rather_than_reporting_all_clear(self, tmp_path: Path) -> None:
        shutil.copytree(MANUSCRIPTS, tmp_path / "manuscripts")
        (tmp_path / "corpus" / "tiny").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            token_collisions(tmp_path, "tiny", ["caprin-heterochromatin"])

    def test_a_direction_token_is_a_phrase_not_a_bare_word(self) -> None:
        planted = _planted()
        for error in planted.errors:
            if error.kind == "effect_direction_flip":
                assert len(error.detection_token.split()) > 1, (
                    "a bare direction word is common enough to appear by accident"
                )


class TestTheRuleIsFrozen:
    def test_the_cue_alternation_is_the_frozen_text(self) -> None:
        """The instrument is frozen before the control runs; this pins the bytes."""
        assert ASSERTION_CUES.pattern == FROZEN_CUES
        assert DETECTION_RULE == "assertion-v1"

    def test_the_dropped_cues_stay_dropped(self) -> None:
        """Each dropped word is ordinary vocabulary here — proven against the real text."""
        for phrase in COLLIDING_PHRASES:
            citation, sentence = _real_manuscript_sentence(phrase)
            assert ASSERTION_CUES.search(sentence) is None, (
                f"{phrase!r} at {citation} matches the cue alternation — a quotation of "
                "committed methods prose could mint an assertion"
            )


class TestTheNegativeControl:
    def test_the_fixture_is_the_committed_record(self) -> None:
        """Re-derive all 230 strings from the committed records and compare, byte for byte.

        Everything except `why` (prose, not derivable) is rebuilt here from `git show` at
        the round-3 commit, so the fixture cannot drift from the records it claims to be.
        """
        fixture = _fixture()
        shows = {record: _git_show(record) for record in ROUND_3_RECORDS}
        missing = [record for record, text in shows.items() if text is None]
        if missing:
            shallow = subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "--is-shallow-repository"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if shallow != "true":
                pytest.fail(
                    f"{missing} are unreachable at {ROUND_3_COMMIT} in a full clone of this "
                    "repository — the negative control has lost its provenance"
                )
            pytest.skip(f"shallow clone: {missing} not fetched")

        errors: dict[str, Any] = {}
        strings: list[dict[str, Any]] = []
        for record in ROUND_3_RECORDS:
            data = json.loads(shows[record] or "")
            errors[record] = data["errors"]
            for result in data["results"]:
                for index, text in enumerate(result["finding_texts"]):
                    strings.append(
                        {
                            "record": record,
                            "system": result["system"],
                            "finding_index": index,
                            "text": text,
                            "credited_for": [
                                error["error_id"]
                                for error in data["errors"]
                                if error["detection_token"].lower() in text.lower()
                            ],
                        }
                    )
        assert len(strings) == 230, len(strings)
        assert sum(1 for s in strings if s["credited_for"]) == 14
        derived = {
            "source_commit": ROUND_3_COMMIT,
            "why": fixture["why"],
            "errors": errors,
            "strings": strings,
        }
        rendered = json.dumps(derived, indent=1, sort_keys=True) + "\n"
        assert FIXTURE.read_bytes() == rendered.encode()

    def test_no_round_3_string_asserts_a_defect(self) -> None:
        """Not one string round 3 scored is credited by the assertion rule."""
        fixture = _fixture()
        strings = fixture["strings"]
        assert len(strings) == 230, len(strings)
        scored = 0
        credited: list[str] = []
        for entry in strings:
            for error in _errors_of(fixture, entry["record"]):
                scored += 1
                if asserts(error, [entry["text"]]):
                    credited.append(
                        f"{entry['record']} {entry['system']} #{entry['finding_index']} "
                        f"vs {error.error_id}: {entry['text'][:160]!r}"
                    )
        assert scored == 690, f"only {scored} string-against-error decisions made"
        assert not credited, "the assertion rule credits round-3 strings:\n" + "\n".join(credited)

    def test_the_control_is_a_floor_not_evidence(self) -> None:
        """No cue occurs in ANY of the 230 strings, so this control cannot fail.

        Measured, not assumed — and the reason the boundary and positive controls below
        are built from real text: they, not this, exercise the same-sentence and
        negation clauses.
        """
        with_a_cue = [
            entry["text"][:120]
            for entry in _fixture()["strings"]
            if ASSERTION_CUES.search(entry["text"])
        ]
        assert not with_a_cue, (
            "some round-3 strings DO contain a cue — the negative control now bites, and "
            f"the rule's behaviour on them is evidence: {with_a_cue}"
        )

    def test_exactly_the_credited_rows_name_a_token(self) -> None:
        """`detect` credits exactly the rows the fixture marks — no more, no fewer."""
        fixture = _fixture()
        rows = 0
        for entry in fixture["strings"]:
            named = [
                error.error_id
                for error in _errors_of(fixture, entry["record"])
                if detect(error, [entry["text"]])
            ]
            assert named == entry["credited_for"], (
                f"{entry['record']} {entry['system']} #{entry['finding_index']}: "
                f"detect says {named}, the record says {entry['credited_for']}"
            )
            rows += 1 if entry["credited_for"] else 0
        assert rows == 14, rows
        assert len({e["text"] for e in fixture["strings"] if e["credited_for"]}) == 13


class TestBoundaryControlsFromRealText:
    def test_a_colliding_phrase_beside_the_token_is_not_an_assertion(self) -> None:
        """Real methods prose naming a planted token scores 0 — no cue fires on it."""
        planted = _planted()
        checked = 0
        for phrase in COLLIDING_PHRASES:
            citation, sentence = _real_manuscript_sentence(phrase)
            for error in planted.errors:
                control = _beside_the_token(sentence, error.detection_token)
                assert len(sentences(control)) == 1, (
                    f"{citation}: the control is not one sentence, so a 0 would be vacuous"
                )
                assert error.detection_token.lower() in control.lower()
                assert phrase.lower() in control.lower()
                assert not asserts(error, [control]), (
                    f"{phrase!r} at {citation} was read as an assertion about "
                    f"{error.detection_token!r}: {control[:200]!r}"
                )
                checked += 1
        assert checked == len(COLLIDING_PHRASES) * len(planted.errors) >= 12, checked


class TestPositiveControlsFromRealText:
    def _real_string(self, credited: bool) -> str:
        fixture = _fixture()
        for entry in fixture["strings"]:
            if entry["record"] == ROUND_3_RECORDS[0] and bool(entry["credited_for"]) is credited:
                return str(entry["text"])
        pytest.fail(f"no round-3 string with credited={credited} to build a control from")

    def test_a_real_string_plus_a_genuine_assertion_is_credited(self) -> None:
        """The quotation earns nothing; the assertion sentence appended to it earns the catch.

        The assertion carries `S. pombe`, whose full stop the shared splitter must not
        read as a sentence end.
        """
        planted = _planted()
        error = next(e for e in planted.errors if e.kind == "gene_symbol_swap")
        real = self._real_string(credited=True)
        assertion = (
            f"The symbol {error.detection_token} here is incorrect; "
            "S. pombe's Dicer is Dcr1."
        )
        assert not asserts(error, [real]), "the record string alone must earn nothing"
        assert asserts(error, [assertion]), "the assertion sentence alone must earn the catch"
        assert asserts(error, [f"{real.strip().removesuffix('.')}. {assertion}"])

    def test_a_fabricated_citation_called_out_by_name_is_credited(self) -> None:
        """The assertion carries `Nat. Metab.` inside a citation the splitter must keep whole."""
        planted = _planted()
        error = next(e for e in planted.errors if e.kind == "fabricated_citation")
        real = self._real_string(credited=False)
        assertion = (
            "The citation (Hollingsworth and Vance, 2019, Nat. Metab. 7:e91188) does not exist."
        )
        assert error.detection_token in assertion
        assert not asserts(error, [real]), "the record string alone must earn nothing"
        assert asserts(error, [assertion]), "the assertion sentence alone must earn the catch"
        assert asserts(error, [f"{real.strip().removesuffix('.')}. {assertion}"])

    def test_a_negated_cue_is_not_an_assertion(self) -> None:
        planted = _planted()
        error = next(e for e in planted.errors if e.kind == "gene_symbol_swap")
        negated = f"Nothing about the {error.detection_token} reference is incorrect."
        assert not asserts(error, [negated]), negated

    def test_the_cue_must_be_in_the_token_s_own_sentence(self) -> None:
        planted = _planted()
        error = next(e for e in planted.errors if e.kind == "gene_symbol_swap")
        split = f"{error.detection_token} is named in this excerpt. That symbol is incorrect."
        assert len(sentences(split)) == 2, "the control is only a control if it is two sentences"
        assert not asserts(error, [split]), split

    def test_a_cue_without_the_token_earns_nothing(self) -> None:
        planted = _planted()
        error = next(e for e in planted.errors if e.kind == "gene_symbol_swap")
        cue_only = "The gene symbol used in this excerpt is incorrect and should read otherwise."
        assert ASSERTION_CUES.search(cue_only), "the control must contain a cue"
        assert error.detection_token not in cue_only
        assert not asserts(error, [cue_only])


class TestTheRulesDocumentedLimits:
    """The two limits the `asserts` docstring discloses, pinned so they stay disclosed."""

    def test_a_negation_anywhere_in_the_clause_suppresses_the_cue(self) -> None:
        """"Dcr2, not Dcr1, is incorrect" is a real assertion and is NOT credited.

        Clause-scoped negation cannot tell a negated claim from a corrected one, so it
        under-credits — the safe direction for a published count, and stated as a limit
        in the docstring rather than hidden.
        """
        planted = _planted()
        error = next(e for e in planted.errors if e.kind == "gene_symbol_swap")
        corrected = f"{error.detection_token}, not Dcr1, is incorrect."
        assert not asserts(error, [corrected]), corrected

    def test_an_unbalanced_parenthesis_falls_back_to_the_plain_splitter(self) -> None:
        """Masking is skipped for unbalanced or nested brackets, with the cost it carries."""
        planted = _planted()
        error = next(e for e in planted.errors if e.kind == "fabricated_citation")
        balanced = (
            "The citation (Hollingsworth and Vance, 2019, Nat. Metab. 7:e91188) does not exist."
        )
        unbalanced = balanced.replace(") does", " does")
        nested = balanced.replace("and Vance, 2019", "and Vance (2019)")
        assert asserts(error, [balanced])
        assert not asserts(error, [unbalanced]), "unbalanced: the plain splitter cuts at Nat."
        assert not asserts(error, [nested]), "nested: the plain splitter cuts at Nat."


class TestScoringCreditsOnlyAssertions:
    def test_restating_the_error_is_not_a_catch(self) -> None:
        """The property, not just the channel: the perturbed sentence itself earns nothing.

        The manuscript sentence carrying the planted token is exactly what round 3 found
        credited on every catch. `detect` still credits it — it is the "named the token"
        upper bound and says so — and `asserts` does not, which is the whole difference
        between the two published numbers.
        """
        planted = _planted()
        checked = 0
        for error in planted.errors:
            quoted = [
                sentence
                for sentence in sentences(planted.text)
                if error.detection_token.lower() in sentence.lower()
            ]
            assert quoted, f"{error.error_id}: the planted token is not in the manuscript text"
            for sentence in quoted:
                assert detect(error, [sentence]), "detect is a substring rule"
                assert not asserts(error, [sentence]), (
                    f"{error.error_id}: quoting the perturbed manuscript was credited as an "
                    f"assertion: {sentence[:200]!r}"
                )
                checked += 1
        assert checked >= 3, checked

    def test_the_harness_only_ever_scores_assertion_channels(self) -> None:
        """A catch requires the token in text the ARM ASSERTED; this pins WHICH texts."""
        from peerpanel.evals import planted_eval

        source = Path(planted_eval.__file__).read_text()
        assert 'v.verdict == "REFUTES"' in source, (
            "the panel's scored channels must exclude non-REFUTES verdicts"
        )
        assert "review.summary" not in source.split("panel_texts")[1][:400], (
            "the converger's prose is not an assertion channel"
        )
