"""The credibility map is bound to the guidance it quotes and to the files it cites.

`CREDIBILITY.md` maps this repository onto the seven steps of the risk-based credibility
assessment framework in the FDA draft guidance saved under `docs/fda/`. Six things could
turn that page into a claim it has not earned, so six things are held here:

- the saved guidance is the document the record says it is (both sha256s);
- the seven section headings are the guidance's own wording, checked against the saved
  text rather than against memory;
- every path the page cites resolves, and at least four steps cite a path that is
  actually committed — a map pointing at files nobody can open is a brochure;
- no SENTENCE on the page gives this system a regulatory verdict, and no claim this repo
  has denied itself appears on it in any form, not even negated;
- the scoring rule the page names is the one the code defines, read from
  `DETECTION_RULE` rather than typed here, and the same for the `eval` target's comment
  in the `Makefile` — the comment a reader meets immediately before running the headline
  measurement;
- what the page SAYS about the regenerate table at the top of `results/RESULTS.md` is
  what that table shows, counts included.

The readiness sweep is scoped to a sentence and then to the verdict's own clause, not to
a line. Round 5 showed why: a line-scoped rule exempted a whole line for a bare "no"
anywhere on it, so "This system is submission-ready today, and no step remains." passed
it. The clause rule is the repository's own — `CLAUSE_BREAKS`, imported from
`peerpanel.evals.planted`, where the identical defect was found and fixed by a pre-run
control before the scoring rule had scored anything.

Every sweep here also proves it can fail: a guard that matches nothing is the defect class
round 3 found, so each rule is run against a control that must trip it.

URLs on that page are never wrapped in backticks — backticks mean "a path in this
repository", and this file resolves every one of them.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from peerpanel.evals import planted
from peerpanel.evals.planted import CLAUSE_BREAKS, DETECTION_RULE
from peerpanel.text.chunks import sentences

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "CREDIBILITY.md"
FDA_DIR = ROOT / "docs" / "fda"
PROVENANCE = FDA_DIR / "PROVENANCE.json"
MAKEFILE = ROOT / "Makefile"
RESULTS = ROOT / "results" / "RESULTS.md"

# The regulatory-verdict vocabulary of requirement 9's readiness rule, as the checklist's
# own grep spells it. A line that matches must also disclaim.
READINESS = re.compile(r"validat|complian|qualif|conform|aligned with|[- ]ready", re.I)
# Requirement 9's own disclaimer words. Held as a tuple so the subset invariant against
# the scoring rule's wider list can iterate them rather than re-parse this pattern.
NEGATION_WORDS = ("not", "never", "no")
NEGATION = re.compile(r"\b(" + "|".join(NEGATION_WORDS) + r")\b", re.I)

FIRST_PERSON = re.compile(r"\b(I|[Mm]y|[Ww]e|[Oo]ur)\b")

BACKTICKED = re.compile(r"`([^`\n]+)`")
PATH_SUFFIXES = (".md", ".py", ".json", ".log", ".txt")
GLOB_CHARS = set("*?[")

_SPEC = json.loads((Path(__file__).parent / "denied_claims.json").read_text())
DENIED_TERMS: set[str] = set(_SPEC["denied"]) | set(_SPEC["denied_everywhere"])

# `assertion-v<n>` wherever prose names the scoring rule. The live id is never typed in
# this file — it is read from the code that defines it — so a bumped rule turns every
# surface still naming the old one red instead of leaving a superseded name standing.
RULE_ID = re.compile(r"assertion-v\d+")

# The two row shapes of the regenerate table at the top of `results/RESULTS.md`, read
# only as far as this page's CLAIM about that table needs.
# `tests/test_regenerate_commands.py` owns the table's own contract; this reads the
# shapes independently rather than importing that file's reading of them, because what
# is being checked here is a claim against the table, not a test against a test.
TABLE_ROW = re.compile(r"^\| (`[^|]+`(?: / `\.log`)?) \| [^|]+ \| ([^|]+) \|$", re.MULTILINE)
HAND_MAINTAINED = "hand-maintained"
RECORD_SUFFIXES = {".json", ".log"}
NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
    20: "twenty",
}

# The three sentences round 5 demonstrated passing the line-scoped guard this file used
# to run: each hands this system a regulatory verdict and carries a bare negation about
# something else later on the same line, which exempted the whole line.
ROUND_FIVE_VERDICTS = (
    "PeerPanel is validated for its context of use, and no reader need take that on trust.",
    "This system is submission-ready today, and no step remains.",
    "The pipeline is compliant with the framework; there is no gap.",
)
# Honest negations, which the rule expects and must keep letting through — the second is
# the page's own closing sentence.
HONEST_NEGATIONS = (
    "This system is not validated for any context of use.",
    "No sentence on this page says this system is validated, compliant or qualified for "
    "anything, because none of that would be true.",
)


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def _provenance() -> dict[str, object]:
    record: dict[str, object] = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    return record


def _saved_text_flat() -> str:
    """The saved guidance with every run of whitespace collapsed.

    The PDF wraps two of the seven headings across two lines, so a heading is only
    contiguous once the whole document is normalised.
    """
    txt = (ROOT / str(_provenance()["file_txt"])).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", txt)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _headings(text: str) -> list[str]:
    return [ln[3:].strip() for ln in text.splitlines() if ln.startswith("## ")]


def _preamble(text: str) -> str:
    """Everything above the first step heading: the title, the scope block, the source."""
    return text.split("\n## ", 1)[0]


def _sections(text: str) -> dict[str, str]:
    """Each step heading mapped to its own body (the closing block belongs to no step)."""
    out: dict[str, str] = {}
    current: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current is not None:
                out[current] = "\n".join(body)
            current, body = line[3:].strip(), []
            continue
        if line.strip() == "---":
            break
        if current is not None:
            body.append(line)
    if current is not None:
        out[current] = "\n".join(body)
    return out


def _cited_paths(text: str) -> list[str]:
    """Backticked strings that name a path: they contain a separator or a known suffix."""
    out = []
    for raw in BACKTICKED.findall(text):
        cited = raw.strip()
        if "/" in cited or cited.endswith(PATH_SUFFIXES):
            out.append(cited)
    return out


def _resolves(cited: str) -> bool:
    target = cited.rstrip("/")
    if set(target) & GLOB_CHARS:
        return any(ROOT.glob(target))
    return (ROOT / target).exists()


def _tracked() -> set[str]:
    listed = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return set(listed)


def _is_committed(cited: str, tracked: set[str]) -> bool:
    target = cited.rstrip("/")
    if set(target) & GLOB_CHARS:
        return any(Path(name).match(target) for name in tracked)
    return target in tracked


def _negated_in_clause(sentence: str, verdict_start: int) -> bool:
    """Does a negation stand between the start of the verdict's clause and the verdict?

    The SCOPE rule is shared, not re-derived: `CLAUSE_BREAKS` is imported from
    `peerpanel.evals.planted`, where a pre-run control defeated a fixed-window version of
    exactly this test and clause scope was recorded as the string-agnostic answer. Sharing
    the constant is what keeps the two from drifting apart.

    The negation VOCABULARY is deliberately not shared. `planted.NEGATION` is the wider
    list (`without`, `nothing`, `unlikely`, `fails to`, …), and a wider list here would
    exempt MORE sentences from a rule whose whole job is to over-flag rather than
    over-exempt. So this keeps requirement 9's own three words, and
    `test_this_guard_is_never_looser_than_the_scoring_rule` holds them a subset of the
    scoring rule's, so the two can only ever differ in the safe direction.
    """
    clause_start = 0
    for break_ in CLAUSE_BREAKS.finditer(sentence, 0, verdict_start):
        clause_start = break_.end()
    return NEGATION.search(sentence[clause_start:verdict_start]) is not None


def _readiness_offenders(text: str) -> list[str]:
    """Sentences that hand out a regulatory verdict without disclaiming it.

    Scope is the SENTENCE — cut with `peerpanel.text.chunks.sentences`, the one splitter
    this repository has, never a second — and then the verdict's own clause inside it. A
    sentence is never read across a line break, which can only make the scope smaller and
    therefore the rule stricter, and the page is written one sentence to a line anyway.

    What this replaces was line-scoped: a line matched the readiness vocabulary and was
    forgiven if a negation token appeared anywhere on it. Round 5 demonstrated three
    verdict sentences walking through that, and all three are controls below.
    """
    offenders: list[str] = []
    for line in text.splitlines():
        for sentence in sentences(line):
            for verdict in READINESS.finditer(sentence):
                if not _negated_in_clause(sentence, verdict.start()):
                    offenders.append(sentence.strip())
                    break
    return offenders


def _denied_hits(text: str) -> list[str]:
    return [
        term for term in sorted(DENIED_TERMS) if re.search(rf"\b{re.escape(term)}\b", text, re.I)
    ]


def _makefile() -> str:
    return MAKEFILE.read_text(encoding="utf-8")


def _stale_rule_ids(text: str) -> list[str]:
    """Every scoring-rule id in this text that is not the one the code defines."""
    return sorted({found for found in RULE_ID.findall(text) if found != DETECTION_RULE})


def _files_named(cell: str) -> list[str]:
    """The record files a table row's first cell names ("`x.json` / `.log`" is two)."""
    names = re.findall(r"`([^`]+)`", cell)
    files = [names[0]]
    for extra in names[1:]:
        files.append(Path(names[0]).with_suffix(extra).name if extra.startswith(".") else extra)
    return files


def _regenerate_split() -> tuple[list[str], list[str]]:
    """The table's records, split: those with a command, those declared hand-maintained."""
    commanded: list[str] = []
    hand: list[str] = []
    for cell, regenerate in TABLE_ROW.findall(RESULTS.read_text(encoding="utf-8")):
        (hand if HAND_MAINTAINED in regenerate else commanded).extend(_files_named(cell))
    return commanded, hand


def _committed_records() -> list[str]:
    return sorted(p.name for p in (ROOT / "results").iterdir() if p.suffix in RECORD_SUFFIXES)


def _regenerate_claim() -> str:
    """The one sentence on the page that describes the regenerate table."""
    claims = [
        line
        for line in _page().splitlines()
        if "regenerat" in line.lower() and "results/RESULTS.md" in line
    ]
    assert len(claims) == 1, f"expected one sentence about the regenerate table, got {claims}"
    return claims[0]


def _claim_defects(claim: str) -> list[str]:
    """Why this sentence does not describe the table `results/RESULTS.md` actually has."""
    commanded, hand = _regenerate_split()
    defects: list[str] = []
    if HAND_MAINTAINED not in claim:
        defects.append(
            "the sentence never uses the table's own word for the rows no command produces, "
            "so it reads as a completeness the table denies"
        )
    counted = (
        ("records under results/", len(_committed_records())),
        ("records whose row names a command", len(commanded)),
        (f"records whose row says {HAND_MAINTAINED}", len(hand)),
    )
    for what, count in counted:
        word = NUMBER_WORDS[count]
        if not re.search(rf"\b{word}\b", claim, re.I):
            defects.append(f"the sentence does not say {word} ({count} {what})")
    return defects


class TestTheSavedGuidanceIsWhatTheRecordSays:
    def test_the_pdf_is_a_pdf_and_matches_its_recorded_sha256(self) -> None:
        record = _provenance()
        pdf = ROOT / str(record["file_pdf"])
        assert pdf.exists(), f"{pdf} is missing — the page quotes a document nobody can open"
        assert pdf.read_bytes()[:4] == b"%PDF", "the saved guidance is not a PDF"
        assert _sha256(pdf) == record["sha256_pdf"]

    def test_the_extracted_text_matches_its_recorded_sha256(self) -> None:
        record = _provenance()
        txt = ROOT / str(record["file_txt"])
        assert txt.exists(), f"{txt} is missing"
        assert len(txt.read_text(encoding="utf-8")) > 20_000, "the extraction looks truncated"
        assert _sha256(txt) == record["sha256_txt"]

    def test_the_record_carries_its_provenance_fields(self) -> None:
        record = _provenance()
        for field in (
            "url",
            "landing_url",
            "fetched_utc",
            "status_at_fetch",
            "docket",
            "extraction_command",
            "public_domain",
        ):
            assert str(record.get(field, "")).strip(), f"PROVENANCE.json has no {field}"
        assert record["docket"] == "FDA-2024-D-4689"
        assert re.fullmatch(r"[0-9a-f]{64}", str(record["sha256_pdf"]))
        assert re.fullmatch(r"[0-9a-f]{64}", str(record["sha256_txt"]))
        assert "pdftotext -layout" in str(record["extraction_command"])
        assert "17 U.S.C" in str(record["public_domain"])


class TestTheSevenHeadingsAreTheGuidancesOwn:
    def test_the_page_carries_exactly_seven_step_headings(self) -> None:
        headings = _headings(_page())
        assert len(headings) == 7, headings
        for i, heading in enumerate(headings, 1):
            assert heading.startswith(f"Step {i}:"), heading

    def test_the_scope_block_comes_first(self) -> None:
        preamble = _preamble(_page())
        assert preamble.startswith("# "), "the page has no title above its first step"
        assert "demonstration system" in preamble
        sentences = [s for s in preamble.split(".\n") if s.strip()]
        assert len(sentences) >= 3, "the scope block is shorter than three sentences"
        assert "docs/fda/PROVENANCE.json" in preamble, "the scope block cites no provenance"

    def test_every_heading_is_verbatim_in_the_saved_text(self) -> None:
        flat = _saved_text_flat()
        assert len(flat) > 20_000
        for heading in _headings(_page()):
            assert re.sub(r"\s+", " ", heading) in flat, f"not the guidance's wording: {heading}"

    def test_the_headings_are_the_ones_the_provenance_extracted(self) -> None:
        steps = _provenance()["steps"]
        assert isinstance(steps, list)
        assert _headings(_page()) == steps

    def test_a_heading_the_guidance_does_not_carry_is_not_found(self) -> None:
        """Non-vacuity: the substring check above can fail."""
        flat = _saved_text_flat()
        assert "Step 8: Determine the Adequacy of the Demonstration" not in flat
        assert "Step 1: Define the Questions of Interest" not in flat


class TestEveryCitedPathResolves:
    def test_the_page_cites_paths_at_all(self) -> None:
        assert len(set(_cited_paths(_page()))) >= 8, "a map that names almost nothing is a claim"

    def test_every_cited_path_exists(self) -> None:
        missing = [cited for cited in set(_cited_paths(_page())) if not _resolves(cited)]
        assert not missing, f"cited but absent from the repository: {missing}"

    def test_a_path_that_does_not_exist_would_be_caught(self) -> None:
        """Non-vacuity: the resolver above says no to something."""
        assert not _resolves("results/there-is-no-such-record.json")
        assert not _resolves("results/planted-eval-*.parquet")

    def test_at_least_four_steps_name_a_committed_path(self) -> None:
        tracked = _tracked()
        assert "README.md" in tracked, "git ls-files returned nothing useful"
        evidenced = {
            heading: [c for c in _cited_paths(body) if _is_committed(c, tracked)]
            for heading, body in _sections(_page()).items()
        }
        named = {h: paths for h, paths in evidenced.items() if paths}
        assert len(named) >= 4, f"only {len(named)} steps name a committed path: {evidenced}"

    def test_a_step_with_no_evidence_says_so(self) -> None:
        tracked = _tracked()
        for heading, body in _sections(_page()).items():
            if any(_is_committed(c, tracked) for c in _cited_paths(body)):
                continue
            assert "not evidenced here — this is a demonstration system" in body.lower(), heading


class TestTheReadinessRuleHolds:
    def test_no_line_hands_this_system_a_regulatory_verdict(self) -> None:
        offenders = _readiness_offenders(_page())
        assert not offenders, f"a verdict without a disclaimer: {offenders}"

    def test_the_sweep_actually_matched_the_page(self) -> None:
        """Non-vacuity: the rule is exercised by real lines, not by their absence."""
        matched = [line for line in _page().splitlines() if READINESS.search(line)]
        assert matched, "no line on the page uses the vocabulary the rule polices"

    def test_a_verdict_sentence_would_be_caught(self) -> None:
        assert _readiness_offenders("This system is validated for its context of use.")
        assert _readiness_offenders("This repository is compliant with the framework.")
        # The readiness vocabulary and the deny list overlap, and a sentence in the
        # overlap is the worst one this page could carry. The term is READ from
        # `denied_claims.json` rather than typed, because this file is itself swept as
        # authored prose: a control that spells the claim out turns the file enforcing
        # the rule into a file that makes the claim — which is the whole reason those
        # terms live in a data file. Every overlapping term is checked, not one.
        readiness_denied = [term for term in sorted(DENIED_TERMS) if READINESS.search(term)]
        assert readiness_denied, "no denied term uses the readiness vocabulary"
        for term in readiness_denied:
            assert _readiness_offenders(f"The pipeline is {term}.")
        assert not _readiness_offenders("This system is not validated for any context of use.")

    def test_the_three_sentences_round_five_demonstrated_are_caught(self) -> None:
        """Permanent controls, kept as the evaluator wrote them.

        Each of the three gives this system a regulatory verdict and then says "no"
        about something else on the same line. The line-scoped rule these replace read
        that "no" as a disclaimer of the verdict and let all three through.
        """
        for sentence in ROUND_FIVE_VERDICTS:
            assert _readiness_offenders(sentence) == [sentence], sentence

    def test_an_honest_negation_still_passes(self) -> None:
        """The other direction: the rule expects disclaimers and must not flag them."""
        for sentence in HONEST_NEGATIONS:
            assert _readiness_offenders(sentence) == [], sentence

    def test_this_guard_is_never_looser_than_the_scoring_rule(self) -> None:
        """The clause SCOPE is shared with `planted`; the negation vocabulary is not.

        This guard keeps requirement 9's three words while the scoring rule reads a wider
        list. Every word this guard treats as a disclaimer must therefore also be one to
        the scoring rule — the subset direction — so that the difference between them can
        only ever make this guard flag more, never less.
        """
        for word in NEGATION_WORDS:
            assert planted.NEGATION.search(word), (
                f"{word!r} exempts a verdict here but is not a negation to the scoring "
                "rule, so this guard is the looser of the two"
            )
        assert CLAUSE_BREAKS is planted.CLAUSE_BREAKS, "the clause scope is no longer shared"


class TestThePageSaysWhatItIs:
    def test_it_calls_itself_a_demonstration_system(self) -> None:
        assert "demonstration system" in _page()

    def test_it_speaks_as_this_repository(self) -> None:
        found = FIRST_PERSON.findall(_page())
        assert not found, f"first person on a repository page: {found}"

    def test_no_denied_claim_appears_anywhere_on_it(self) -> None:
        assert _denied_hits(_page()) == []

    def test_the_denied_sweep_can_fire(self) -> None:
        """Non-vacuity: every term in the list is one this matcher would find."""
        assert DENIED_TERMS
        control = " ; ".join(sorted(DENIED_TERMS))
        assert _denied_hits(control) == sorted(DENIED_TERMS)


class TestThePageNamesTheRuleTheCodeDefines:
    """The scoring rule's id is read from the code, never typed on the page.

    `DETECTION_RULE` is bumped whenever WHAT the rule scores changes, for the reason the
    code states beside it: so that no record can carry an old rule name over a new count.
    Round 5 found all three evaluators reading a superseded id off `CREDIBILITY.md` — in
    the clause asserting that the name is the one in the code and in every record — and
    off the `eval` target's comment, which a reader meets immediately before running the
    headline measurement. Both surfaces are swept for any `assertion-v<n>`, and every hit
    must be the live one.

    `DECISIONS.md` is deliberately out of scope: it is a dated log, and an entry naming
    the id it introduced is correct history, superseded in that same file by the entry
    recording the bump.
    """

    def test_the_page_names_the_live_rule_id(self) -> None:
        page = _page()
        assert DETECTION_RULE in page, (
            f"CREDIBILITY.md names no scoring rule; the code defines {DETECTION_RULE}"
        )
        assert _stale_rule_ids(page) == [], (
            f"CREDIBILITY.md carries {_stale_rule_ids(page)}; the code defines {DETECTION_RULE}"
        )

    def test_the_makefile_comment_names_the_live_rule_id(self) -> None:
        makefile = _makefile()
        assert DETECTION_RULE in makefile, (
            f"the Makefile comment names no scoring rule; the code defines {DETECTION_RULE}"
        )
        assert _stale_rule_ids(makefile) == [], (
            f"the Makefile carries {_stale_rule_ids(makefile)}; the code defines "
            f"{DETECTION_RULE}, and the comment on the `eval` target is read immediately "
            "before the headline measurement is run"
        )

    def test_a_superseded_id_would_be_caught(self) -> None:
        """Non-vacuity, with the superseded id derived from the live one, not typed."""
        version = int(DETECTION_RULE.rsplit("v", 1)[1])
        superseded = f"assertion-v{version - 1}"
        assert superseded != DETECTION_RULE
        assert _stale_rule_ids(f"the rule that scored it — `{superseded}`") == [superseded]
        assert _stale_rule_ids(f"the rule that scored it — `{DETECTION_RULE}`") == []


class TestWhatThePageSaysAboutTheRegenerateTable:
    """The page's claim about `results/RESULTS.md` is read back from that table.

    Round 5: the page claimed every record under `results/` names the command that
    regenerates it, and the cited table says the opposite of three of them in terms —
    they are hand-maintained, because no command produces them and inventing one would be
    a reproduction instruction that does not reproduce. A citation the cited file refutes
    is the error this page can least afford, so the counts the page gives are checked
    against the table and against the directory.
    """

    def test_the_table_is_read_in_both_shapes(self) -> None:
        commanded, hand = _regenerate_split()
        assert commanded, "no row of the table names a command — the parse is reading nothing"
        assert hand, (
            "no row declares itself hand-maintained, so the page's exception is stale; "
            "say so on the page rather than leaving a clause nothing corresponds to"
        )

    def test_the_table_accounts_for_every_committed_record(self) -> None:
        commanded, hand = _regenerate_split()
        assert sorted(commanded + hand) == _committed_records(), (
            "the page says the table accounts for every record under results/; it does not"
        )

    def test_the_page_describes_the_table_the_repository_has(self) -> None:
        claim = _regenerate_claim()
        assert _claim_defects(claim) == [], (
            f"CREDIBILITY.md's sentence about the regenerate table is wrong about it: "
            f"{_claim_defects(claim)}. The sentence reads: {claim}"
        )

    def test_the_universal_claim_round_five_found_would_be_caught(self) -> None:
        """Non-vacuity: the exact sentence the evaluators found must not pass."""
        stale = (
            "Every record under `results/` names the one command that regenerates it, "
            "in the table at the top of `results/RESULTS.md`."
        )
        assert _claim_defects(stale), "the guard would have passed the sentence round 5 found"

