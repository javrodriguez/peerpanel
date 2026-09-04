"""The credibility map is bound to the guidance it quotes and to the files it cites.

`CREDIBILITY.md` maps this repository onto the seven steps of the risk-based credibility
assessment framework in the FDA draft guidance saved under `docs/fda/`. Four things could
turn that page into a claim it has not earned, so four things are held here:

- the saved guidance is the document the record says it is (both sha256s);
- the seven section headings are the guidance's own wording, checked against the saved
  text rather than against memory;
- every path the page cites resolves, and at least four steps cite a path that is
  actually committed — a map pointing at files nobody can open is a brochure;
- no sentence on the page gives this system a regulatory verdict, and no claim this repo
  has denied itself appears on it in any form, not even negated.

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

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "CREDIBILITY.md"
FDA_DIR = ROOT / "docs" / "fda"
PROVENANCE = FDA_DIR / "PROVENANCE.json"

# The regulatory-verdict vocabulary of requirement 9's readiness rule, as the checklist's
# own grep spells it. A line that matches must also disclaim.
READINESS = re.compile(r"validat|complian|qualif|conform|aligned with|[- ]ready", re.I)
NEGATION = re.compile(r"\b(not|never|no)\b", re.I)

FIRST_PERSON = re.compile(r"\b(I|[Mm]y|[Ww]e|[Oo]ur)\b")

BACKTICKED = re.compile(r"`([^`\n]+)`")
PATH_SUFFIXES = (".md", ".py", ".json", ".log", ".txt")
GLOB_CHARS = set("*?[")

_SPEC = json.loads((Path(__file__).parent / "denied_claims.json").read_text())
DENIED_TERMS: set[str] = set(_SPEC["denied"]) | set(_SPEC["denied_everywhere"])


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


def _readiness_offenders(text: str) -> list[str]:
    """Lines that hand out a regulatory verdict without disclaiming it."""
    return [
        line for line in text.splitlines() if READINESS.search(line) and not NEGATION.search(line)
    ]


def _denied_hits(text: str) -> list[str]:
    return [
        term for term in sorted(DENIED_TERMS) if re.search(rf"\b{re.escape(term)}\b", text, re.I)
    ]


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
