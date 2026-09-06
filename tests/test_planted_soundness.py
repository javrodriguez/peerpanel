"""The planted-error benchmark must be able to earn the score it reports.

Six ways a planted-error harness can report a number that means nothing, all
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
5. Close one of quotation's two routes and publish that both are closed. Round 3
   made `scored_text` drop a finding's `quote` field; nothing stopped a model
   putting manuscript text in `text`, and round 4 measured that 147 of the 235
   committed scored strings (63%) are verbatim slices of the perturbed manuscript,
   including every string that ever earned a `named token` credit. `own_prose` is
   the second route closed, and `TestOnlyOwnProseIsScored` below is its control.
6. Close that route with a test a full stop defeats. `own_prose` dropped a sentence
   only when it was a substring of the manuscript, so a clause copied out of a
   paragraph and terminated with a period survived into the scored surface. Round 5
   measured it: 68% rather than 59% of the characters written back are manuscript
   slices, one arm's published "own prose 41%" is really about 16%, and two of the
   five `named token` credits on the first screen existed only because a copied
   sentence ended in a period. `TestTerminalPunctuationIsTolerated` is its control.

These tests pin all six as properties of the harness, so a future change that
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
import re
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
    own_prose,
    plant_errors,
)
from peerpanel.manuscripts.store import read_manuscript
from peerpanel.text.chunks import sentences

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPTS = ROOT / "manuscripts"
RESULTS = ROOT / "results"
PLANTED_RECORDS = sorted(RESULTS.glob("planted-eval-*.json"))
MET17_RECORD = RESULTS / "planted-eval-met17-auxotroph.json"
# The exact string both round-4 evaluators quote as the one that minted the panel's
# `named token` credit on met17. It is asserted to be IN the committed record before it
# is used, so this control is the repository's own text and not a reconstruction of it.
# The reference ranges carry an EN DASH, written as an escape so the pin says which dash
# it is rather than leaving a reader to tell two glyphs apart.
CITED_MET17_QUOTATION = (
    "The MET18 gene, also known as MET15 or MET25 [13\u201315], catalyzes homocysteine "
    "synthesis by reacting H2S with O-acetyl homoserine (i.e. displaying OAH "
    "sulfhydrylase activity) [16\u201318]."
)
# Word characters minus the underscore: "met17\u0394" is one word here, and case-folding
# turns its delta lowercase, so a hand-written a-z class would silently drop it.
_WORD = re.compile(r"[^\W_]+")
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


def _normalised(text: str) -> str:
    """The quotation test's string shape, written out here rather than imported.

    `own_prose` normalises whitespace and case-folds before testing for a substring;
    this is that rule stated independently, so a test comparing against it is a check
    and not a tautology.
    """
    return re.sub(r"\s+", " ", text).strip().casefold()


# The two punctuation runs `own_prose` takes off a sentence before the substring test,
# written out here rather than imported for the same reason `_normalised` above is: a
# control that borrowed the implementation's own characters would agree with it by
# construction and could never catch the set drifting.
TOLERATED_LEADING = "\"'([ "
TOLERATED_TRAILING = ".,;:!?\"')] "


def _trimmed(text: str) -> str:
    """`_normalised`, then the punctuation `own_prose` tolerates taken off the ends."""
    return _normalised(text).lstrip(TOLERATED_LEADING).rstrip(TOLERATED_TRAILING)


def _record_and_manuscript(path: Path) -> tuple[dict[str, Any], str]:
    """A committed planted-eval record beside the PERTURBED text its arms were shown.

    The manuscript is re-planted here rather than assumed, and the planted errors are
    compared with the record's own: if they differ, the text being tested for quotation
    is not the text that produced these strings and every count below would be measuring
    the wrong pair.
    """
    record: dict[str, Any] = json.loads(path.read_text())
    source = MANUSCRIPTS / str(record["manuscript"])
    _header, body = read_manuscript(source)
    planted = plant_errors(body, source.name, window_words=EXCERPT_WORDS)
    assert [error.model_dump() for error in planted.errors] == record["errors"], (
        f"{path.name}: re-planting does not reproduce the record's errors — the perturbed "
        "text compared against here is not the text its arms were shown"
    )
    return record, planted.text


def _fixture() -> dict[str, Any]:
    if not FIXTURE.exists():  # pragma: no cover - the fixture is committed
        pytest.fail(f"{FIXTURE} is missing — the negative control cannot run without it")
    fixture: dict[str, Any] = json.loads(FIXTURE.read_text())
    return fixture


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
        """The instrument is frozen before the control runs; this pins the bytes.

        The rule id moved v1 -> v2 when `own_prose` changed WHAT is scored; the
        alternation it is judged with did not move, and that is what this pins. A cue
        list edited to make a control pass would fail here whatever the id says.
        """
        assert ASSERTION_CUES.pattern == FROZEN_CUES
        assert DETECTION_RULE == "assertion-v2"

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


class TestOnlyOwnProseIsScored:
    """Round 4 (report 1 finding 2, report 3 finding 3): `text` was quotation's other route.

    `scored_text` drops a finding's `quote`, and the repository published that as "only
    the finding's own prose is scored". It was not: on met17 all 8 of the panel methods
    reviewer's findings have `text` byte-identical to their own `quote`, and 63% of every
    committed scored string is a verbatim slice of the perturbed manuscript. `own_prose`
    removes those sentences before either rule sees them.

    Every string in this class comes out of a committed record or a committed manuscript.
    The one assertion sentence that is written here is written to be scored, not to be
    quoted, and it is the only invented text in the class.
    """

    def _met17(self) -> tuple[dict[str, Any], str]:
        if not MET17_RECORD.exists():  # pragma: no cover - the record is committed
            pytest.fail(f"{MET17_RECORD} is missing — these controls have no source text")
        return _record_and_manuscript(MET17_RECORD)

    def _pure_quotation(self, record: dict[str, Any], perturbed: str) -> str:
        """The first committed string of this record that is quotation and nothing else."""
        for result in record["results"]:
            for text in result["finding_texts"]:
                if not own_prose(str(text), perturbed):
                    return str(text)
        pytest.fail(
            "no committed string is pure quotation — this control would have to be "
            "invented, and the defect it guards would be unmeasured"
        )

    def test_a_committed_pure_quotation_string_is_scored_on_nothing(self) -> None:
        """A string that is only manuscript leaves no residue, so neither rule can fire."""
        record, perturbed = self._met17()
        quotation = self._pure_quotation(record, perturbed)
        assert _normalised(quotation) in _normalised(perturbed), (
            "the control string is not actually quotation"
        )
        residue = own_prose(quotation, perturbed)
        assert residue == "", residue
        for error in (PlantedError(**error) for error in record["errors"]):
            assert not asserts(error, [residue]), f"{error.error_id}: {quotation[:160]!r}"
            assert not detect(error, [residue]), f"{error.error_id}: {quotation[:160]!r}"

    def test_an_assertion_appended_to_that_quotation_survives_and_scores(self) -> None:
        """The stripping is surgical: the quotation goes, the arm's own claim stays.

        Without this the change would be indistinguishable from scoring nothing at all,
        which would also make the headline 0.
        """
        record, perturbed = self._met17()
        quotation = self._pure_quotation(record, perturbed)
        error = next(
            PlantedError(**error)
            for error in record["errors"]
            if error["kind"] == "gene_symbol_swap"
        )
        assertion = (
            f"The symbol {error.detection_token} here is incorrect; "
            "the S. cerevisiae gene is MET17."
        )
        assert _normalised(assertion) not in _normalised(perturbed), (
            "the assertion is itself manuscript text — it would be stripped for the "
            "right reason and prove nothing"
        )
        combined = f"{quotation.strip()} {assertion}"
        assert len(sentences(combined)) > len(sentences(quotation)), (
            "the assertion did not become a sentence of its own — the control is vacuous"
        )
        residue = own_prose(combined, perturbed)
        assert residue == assertion, residue
        assert asserts(error, [residue]), residue
        assert detect(error, [residue]), residue
        assert asserts(error, [combined]), (
            "v2 credits a string v1 did not — the change must only ever REMOVE candidates"
        )

    def test_the_met17_credit_the_evaluators_cite_is_no_longer_minted(self) -> None:
        """The specific published cell round 4 named: `named token` 1/3, minted by a quote."""
        record, perturbed = self._met17()
        panel = next(result for result in record["results"] if result["system"] == "panel")
        assert CITED_MET17_QUOTATION in panel["finding_texts"], (
            "the cited string is not in the committed panel record — this control has "
            "lost its provenance and would be testing invented text"
        )
        error = next(
            PlantedError(**error)
            for error in record["errors"]
            if error["detection_token"] == "MET18"
        )
        assert detect(error, [CITED_MET17_QUOTATION]), (
            "the defect itself: this string is what credited the panel under v1"
        )
        residue = own_prose(CITED_MET17_QUOTATION, perturbed)
        assert residue == "", residue
        assert not detect(error, [residue]), "the quotation still mints the named-token credit"
        assert not asserts(error, [residue])

    def test_sharing_the_manuscripts_vocabulary_is_not_quotation(self) -> None:
        """The stripping is a substring test, not a similarity one — so it is not over-eager.

        The survivor is chosen as the committed string with the HIGHEST share of
        manuscript vocabulary that `own_prose` keeps whole: if the rule were fitting on
        resemblance rather than identity, this is the string it would take first.
        """
        record, perturbed = self._met17()
        vocabulary = set(_WORD.findall(perturbed.casefold()))
        assert vocabulary, "the perturbed manuscript produced no vocabulary to share"
        best: tuple[float, str] | None = None
        for result in record["results"]:
            for text in result["finding_texts"]:
                if _normalised(own_prose(str(text), perturbed)) != _normalised(str(text)):
                    continue
                words = _WORD.findall(str(text).casefold())
                if len(words) < 20:
                    continue
                share = sum(1 for word in words if word in vocabulary) / len(words)
                if best is None or share > best[0]:
                    best = (share, str(text))
        assert best is not None, "every committed string was stripped — the rule is over-eager"
        share, survivor = best
        assert _normalised(survivor) not in _normalised(perturbed), survivor[:160]
        assert share >= 0.8, (
            f"the most manuscript-flavoured survivor shares only {share:.0%} of its words "
            "with the manuscript — this control no longer proves the rule is not "
            f"resemblance-based: {survivor[:160]!r}"
        )
        assert _normalised(own_prose(survivor, perturbed)) == _normalised(survivor), (
            f"a string sharing {share:.0%} of the manuscript's vocabulary but quoting none "
            f"of its sentences was stripped: {survivor[:200]!r}"
        )

    def test_stripping_quotation_can_only_lower_a_count(self) -> None:
        """D-11's whole argument, measured rather than asserted: v2 never credits more.

        A scoring change made after the results are known is a tune unless it is
        arithmetically incapable of flattering the headline. Over every (planted error,
        committed string) pair in both records, and for BOTH rules, a credit on the
        `own_prose` residue must imply a credit on the raw string. A rule change that
        could raise a published number fails here.
        """
        pairs = 0
        raised: list[str] = []
        for path in PLANTED_RECORDS:
            record, perturbed = _record_and_manuscript(path)
            errors = [PlantedError(**error) for error in record["errors"]]
            for result in record["results"]:
                for text in (str(text) for text in result["finding_texts"]):
                    residue = own_prose(text, perturbed)
                    for error in errors:
                        pairs += 1
                        for rule in (asserts, detect):
                            if rule(error, [residue]) and not rule(error, [text]):
                                raised.append(
                                    f"{path.name} {result['system']} {error.error_id} "
                                    f"{rule.__name__}: {text[:120]!r}"
                                )
        assert pairs > 0, "the sweep compared nothing"
        assert not raised, "own_prose RAISED a credit:\n" + "\n".join(raised)

    def test_how_much_of_the_committed_scored_surface_is_pure_quotation(self) -> None:
        """The measurement, per arm, over every committed record — the numbers to publish.

        Both measures, because the repository publishes both: the share of STRINGS that
        leave no own prose at all, and the share of CHARACTERS written back that are
        manuscript slices — which is the figure `results/RESULTS.md` puts in bold and the
        per-arm "own prose, in characters" column it puts beside it. Reported per arm in
        the assertion message and on stdout so every published figure is read off a test
        rather than off a notebook nobody kept.

        Asserted non-zero so it cannot pass on an empty sweep. The numbers move with the
        run — they are a property of what the models wrote — so nothing here pins a
        value; what is pinned is that the measurement exists and is per arm.
        """
        assert PLANTED_RECORDS, f"no planted-eval records under {RESULTS} to measure"
        rows: list[str] = []
        pure_total = 0
        string_total = 0
        own_chars_total = 0
        chars_total = 0
        for path in PLANTED_RECORDS:
            record, perturbed = _record_and_manuscript(path)
            subject = path.stem.removeprefix("planted-eval-")
            for result in record["results"]:
                texts = [str(text) for text in result["finding_texts"]]
                residues = [own_prose(text, perturbed) for text in texts]
                pure = [residue for residue in residues if not residue]
                own_chars = sum(len(residue) for residue in residues)
                chars = sum(len(text.strip()) for text in texts)
                own_share = own_chars / chars if chars else 0.0
                rows.append(
                    f"  {subject} · {result['system']}: {len(pure)}/{len(texts)} strings "
                    f"are pure quotation · own prose {own_chars}/{chars} characters "
                    f"({own_share:.0%}), so {1 - own_share:.0%} of what it wrote back "
                    "is manuscript"
                )
                pure_total += len(pure)
                string_total += len(texts)
                own_chars_total += own_chars
                chars_total += chars
        assert string_total, "the sweep read no strings"
        assert chars_total, "the sweep read no characters"
        share = pure_total / string_total
        quoted_chars = chars_total - own_chars_total
        char_share = quoted_chars / chars_total
        report = (
            f"pure quotation on the committed records: {pure_total}/{string_total} "
            f"({share:.0%}) of scored strings leave no own prose; "
            f"{quoted_chars}/{chars_total} ({char_share:.0%}) of the characters written "
            "back are slices of the manuscript\n" + "\n".join(rows)
        )
        print(report)
        assert pure_total > 0, (
            "not one committed string is pure quotation — either the records were "
            f"regenerated or the sweep is measuring nothing:\n{report}"
        )
        assert quoted_chars > 0, (
            f"not one character of the committed surface is quotation:\n{report}"
        )


class TestTerminalPunctuationIsTolerated:
    """Round 5 (report 2 finding 2): a full stop defeated the quotation strip.

    `own_prose` dropped a sentence only when its normalised form was a SUBSTRING of the
    perturbed manuscript, so a clause copied out of the middle of a paragraph and
    terminated with a period was not a substring and survived into the scored surface.
    Twenty-four sentences across the two committed records are in that class, and they
    carried published numbers: the manuscript share of the characters written back reads
    68% rather than 59%, one arm's published "own prose 41%" on caprin is really about
    16%, and the `named token` bound falls from 2/3 to 1/3 for both single-agent arms on
    caprin. The `asserted` headline is 0 either way.

    Every string here is derived from the committed records and manuscripts rather than
    written out as a literal, because the records are regenerated whenever what is scored
    changes and a pinned literal would then be testing text no arm wrote. A derivation
    that finds nothing fails loudly rather than passing on an empty sweep.
    """

    def _saved_by_punctuation(self) -> tuple[Path, dict[str, Any], str, str]:
        """A committed one-sentence string that is quotation ONLY once the ends are trimmed.

        Returns the record's path, the record, the perturbed manuscript and the string.
        One sentence on purpose: the controls below add words to it and read the residue
        back, which is only unambiguous when the whole string is the sentence.
        """
        for path in PLANTED_RECORDS:
            record, perturbed = _record_and_manuscript(path)
            body = _normalised(perturbed)
            for result in record["results"]:
                for text in (str(text) for text in result["finding_texts"]):
                    if len(sentences(text)) != 1:
                        continue
                    if _normalised(text) in body or _trimmed(text) not in body:
                        continue
                    return path, record, perturbed, text
        pytest.fail(
            "no committed string is quotation saved by its punctuation — either the "
            "records were regenerated by an arm that never copies a terminated clause, "
            "or this control is measuring nothing"
        )

    def test_a_committed_quotation_ending_in_punctuation_is_scored_on_nothing(self) -> None:
        """The defect itself, on the repository's own text: it survived, it no longer does."""
        path, record, perturbed, quotation = self._saved_by_punctuation()
        assert _normalised(quotation) not in _normalised(perturbed), (
            f"{path.name}: this string is a plain substring of the manuscript, so it was "
            "already dropped and proves nothing about punctuation"
        )
        assert _trimmed(quotation) in _normalised(perturbed), (
            f"{path.name}: this string is not quotation once trimmed — the control has "
            "lost its subject"
        )
        residue = own_prose(quotation, perturbed)
        assert residue == "", (
            f"{path.name}: a copied sentence still reaches the scored surface because it "
            f"ends in punctuation: {quotation[:200]!r} -> {residue[:200]!r}"
        )
        for error in (PlantedError(**error) for error in record["errors"]):
            assert not asserts(error, [residue]), f"{error.error_id}: {quotation[:160]!r}"
            assert not detect(error, [residue]), f"{error.error_id}: {quotation[:160]!r}"

    def test_a_quotation_wrapped_in_quotation_marks_is_recognised(self) -> None:
        """The leading half of the strip, pinned on a real string rather than left dormant.

        No committed sentence starts with a quotation mark today — every one of the
        twenty-four is terminal punctuation alone — so this control puts a committed
        quotation in the marks a model uses when it says it is quoting, and nothing else.
        """
        _path, _record, perturbed, quotation = self._saved_by_punctuation()
        quoted = f'"{quotation.strip()}"'
        assert _normalised(quoted) not in _normalised(perturbed), (
            "the wrapped string is a plain substring — the control is vacuous"
        )
        assert own_prose(quoted, perturbed) == "", own_prose(quoted, perturbed)

    def test_adding_words_to_a_quotation_still_survives_with_its_own_prose(self) -> None:
        """The tolerance strips punctuation, never words — the over-eagerness control.

        The same committed quotation with words of the arm's own added: a clause after
        it, a single word after it, and a single word in front of it. None of the three
        is manuscript any more, trimmed or not, so each must survive WHOLE — the rule is
        still a substring test on a whole sentence and did not become a resemblance test
        that swallows the added words along with the copied ones.

        The one-word variants are the ones that carry the weight. A strip that ate a word
        as well as the punctuation would still pass a control that added a whole clause,
        because the clause keeps the sentence off the manuscript by itself; it cannot
        pass these, where the single added word is the only thing making the sentence the
        arm's own.
        """
        _path, _record, perturbed, quotation = self._saved_by_punctuation()
        opening = quotation.strip().rstrip(TOLERATED_TRAILING)
        for added in (
            f"{opening}, which this excerpt never shows.",
            f"{opening} everywhere.",
            f"Notably, {opening}.",
        ):
            assert len(sentences(added)) == 1, (
                "the addition split into more than one sentence — the control no longer "
                f"tests a single sentence: {added[:200]!r}"
            )
            assert _trimmed(added) not in _normalised(perturbed), (
                f"the addition is itself manuscript text — {added[:200]!r} would be "
                "dropped for the right reason and prove nothing"
            )
            residue = own_prose(added, perturbed)
            assert residue == added.strip(), (
                "a sentence that quotes the manuscript AND adds words of its own lost "
                f"them: {added[:200]!r} -> {residue[:200]!r}"
            )

    def test_a_short_assertion_is_not_swallowed_by_the_strip(self) -> None:
        """A short, punctuated, quote-wrapped assertion keeps every word and still scores.

        The strip takes runs of punctuation off the ends, so the shape most at risk of
        being trimmed to nothing is the shortest real assertion this benchmark can be
        given: the planted token and a cue, in quotation marks, ending in a period. It
        must survive whole and be credited by both rules — otherwise the repair would be
        removing detections rather than removing quotation.
        """
        _path, record, perturbed, _quotation = self._saved_by_punctuation()
        error = next(
            PlantedError(**error)
            for error in record["errors"]
            if error["kind"] == "gene_symbol_swap"
        )
        assertion = f'"{error.detection_token} is incorrect."'
        assert _trimmed(assertion), "the strip reduced a real assertion to nothing"
        assert _trimmed(assertion) not in _normalised(perturbed), (
            f"{assertion!r} is itself a slice of the manuscript — this control cannot "
            "tell an over-eager strip from the substring rule working"
        )
        residue = own_prose(assertion, perturbed)
        assert residue == assertion, f"{assertion!r} -> {residue!r}"
        assert asserts(error, [residue]), residue
        assert detect(error, [residue]), residue
