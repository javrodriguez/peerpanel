"""Invariant sweeps over AUTHORED PROSE.

The absolutes this repo claims about itself have to hold everywhere they could
be violated, not only where they are expected to. Two rules shape the sweep:

- Only prose the repo AUTHORS is swept: markdown, docstrings, commit subjects.
- Captured material is EVIDENCE, not authorship: model output under `results/`,
  and under `docs/fda/` a federal guidance document saved verbatim beside its
  sha256. A deny-listed word appearing in a genuine capture must never be
  "fixed", since the only way to do that is to edit a recording — and editing
  the saved guidance would break the hash that proves it is the guidance. Those
  surfaces are checked for the one absolute that would be a lie about the system
  itself, wherever it appeared.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The terms live in a DATA file, deliberately. Holding them as prose inside the
# test that enforces them meant this file tripped its own sweep, which was
# answered with a whole-file exemption — and that exemption silently covered the
# one absolute too. Data cannot make a claim.
_SPEC = json.loads((Path(__file__).parent / "denied_claims.json").read_text())
DENIED: dict[str, str] = _SPEC["denied"]
DENIED_EVERYWHERE: set[str] = set(_SPEC["denied_everywhere"])

# `.diff` is here because `prepared-diffs/` holds authored prose in a diff's clothing:
# staged cross-link lines that will one day appear on two public pages. A deny-listed
# claim is no less a claim for sitting behind a `+`, and the flip is exactly the moment
# nobody would re-read them.
AUTHORED_SUFFIXES = {".md", ".py", ".toml", ".yml", ".yaml", ".diff"}
# Directories whose contents are CAPTURED rather than written: model output under
# `results/`, and under `docs/fda/` a federal guidance document saved verbatim with its
# sha256. Neither may be edited to pass a sweep — the only way to "fix" a word in a
# recording is to falsify it, and the saved guidance would stop matching its own
# recorded hash. Both are still checked for the one absolute (DENIED_EVERYWHERE), which
# would be a lie about this system wherever it appeared. Paths are relative to the repo
# root, so a nested directory can be named.
EVIDENCE_DIRS = {"results", "docs/fda"}
# Captured text formats. The saved PDF is skipped: it is not text, and its integrity is
# proven by sha256 in docs/fda/PROVENANCE.json rather than by reading it.
EVIDENCE_SUFFIXES = {".log", ".txt", ".json"}


def _in_evidence_dir(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return any(relative.is_relative_to(Path(d)) for d in EVIDENCE_DIRS)


def _is_authored(path: Path) -> bool:
    """Is this file prose this repository WROTE?

    A `.md` inside an evidence directory is still authored — `results/RESULTS.md` is
    the reading of the records, not a record — but the captures beside it are not.
    """
    if path.suffix not in AUTHORED_SUFFIXES:
        return False
    return not (_in_evidence_dir(path) and path.suffix != ".md")


def _authored_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return [
        path
        for name in listed
        if (path := ROOT / name).exists() and _is_authored(path)
    ]


DISCLAIMERS = ("never", "not ", "avoid", "does not", "must not", "forbidden", "denied")


def _disclaimed(text: str, term: str) -> bool:
    """Does every line mentioning this term also disclaim it?"""
    lines = [ln for ln in text.splitlines() if re.search(rf"\b{re.escape(term)}\b", ln, re.I)]
    return bool(lines) and all(any(d in ln.lower() for d in DISCLAIMERS) for ln in lines)


def _hits(text: str, terms: dict[str, str] | set[str]) -> list[str]:
    found = []
    for term in terms:
        if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
            found.append(term)
    return found


class TestAuthoredProse:
    def test_no_denied_claims_in_authored_files(self) -> None:
        offenders: dict[str, list[str]] = {}
        for path in _authored_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            hits = _hits(text, DENIED)
            # LIMITATIONS names some forbidden phrases in order to disclaim them,
            # which is the opposite of claiming them. The exemption used to be
            # whole-file and covered the one absolute too; it is now narrow — a
            # denied term is forgiven ONLY where every line mentioning it also
            # disclaims it, and nothing is ever exempt from DENIED_EVERYWHERE.
            if path.name == "LIMITATIONS.md":
                hits = [
                    term
                    for term in hits
                    if term in DENIED_EVERYWHERE or not _disclaimed(text, term)
                ]
            if hits:
                offenders[str(path.relative_to(ROOT))] = hits
        assert not offenders, offenders

    def test_no_denied_claims_in_commit_messages(self) -> None:
        log = subprocess.run(
            ["git", "log", "--format=%s"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
        assert not _hits(log, DENIED)

    def test_captured_evidence_is_never_edited_to_pass(self) -> None:
        """Captures are checked for the one absolute only — never groomed.

        The sweep asserts it actually swept something, PER DIRECTORY. An earlier
        version globbed `demo/`, a directory that does not exist, so half of this test
        had never checked a byte — the same shape as every other defect this repo has
        caught: a check that cannot fail. A total count would hide exactly that again
        the moment a second evidence directory was added.
        """
        for directory in sorted(EVIDENCE_DIRS):
            root = ROOT / directory
            assert root.is_dir(), f"{directory}/ does not exist — this sweep would check nothing"
            swept = 0
            for path in root.glob("**/*"):
                if path.is_file() and path.suffix in EVIDENCE_SUFFIXES:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    assert not _hits(text, DENIED_EVERYWHERE), path
                    swept += 1
            assert swept, f"no captured evidence found to sweep in {directory}/"

    def test_the_credibility_map_is_swept_as_authored_prose(self) -> None:
        """The FDA page sits beside the saved guidance, and only one of the two is
        this repository's own words. `docs/fda/` is evidence — the government text is
        quoted, never edited — while `CREDIBILITY.md` is authored and therefore swept
        in full, with no exemption of its own: the narrow disclaimer exemption is
        keyed to LIMITATIONS.md alone, and requirement 9 wants this page to carry no
        denied term even in negation.
        """
        page = ROOT / "CREDIBILITY.md"
        assert page.is_file(), "CREDIBILITY.md is missing — it is requirement 9's page"
        assert _is_authored(page), (
            "CREDIBILITY.md is not classified as authored prose, so the deny-list sweep "
            "would never read the one page written in a regulator's vocabulary"
        )
        assert not _in_evidence_dir(page), "CREDIBILITY.md is this repo's words, not a capture"
        assert not _hits(page.read_text(), DENIED), (
            f"CREDIBILITY.md carries {_hits(page.read_text(), DENIED)}; the page holds no "
            "exemption — only LIMITATIONS.md does, and only for a line that disclaims"
        )
        saved = sorted((ROOT / "docs" / "fda").glob("*.txt"))
        assert saved, "docs/fda/ carries no saved guidance text for the page to quote"
        for path in saved:
            assert not _is_authored(path), f"{path.name} is a capture, not this repo's prose"


class TestRepoBoundaries:
    def test_gauntlet_is_never_tracked(self) -> None:
        listed = subprocess.run(
            ["git", "ls-files", "gauntlet"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        assert listed == "", "the evaluator record must never enter the public repo"

    def test_every_committed_corpus_doc_is_attributed(self) -> None:
        from peerpanel.corpus.models import CorpusManifest

        citations = (ROOT / "CITATIONS.md").read_text()
        for doc in CorpusManifest.load(ROOT / "corpus" / "ci.manifest.json").docs:
            assert f"**Pinned:** {doc.pmcid}.{doc.version}" in citations

    def test_results_are_actually_published(self) -> None:
        """The repo's whole positioning is that the measurements are visible."""
        listed = subprocess.run(
            ["git", "ls-files", "results"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
        names = {Path(p).name for p in listed}
        assert {"RESULTS.md", "ablation-demo.json", "build-stats-demo.json"} <= names
