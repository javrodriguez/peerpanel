"""Invariant sweeps over AUTHORED PROSE.

The absolutes this repo claims about itself have to hold everywhere they could
be violated, not only where they are expected to. Two rules shape the sweep:

- Only prose the repo AUTHORS is swept: markdown, docstrings, commit subjects.
- Captured model output (`results/`) is EVIDENCE, not authorship.
  A deny-listed word appearing in a genuine capture must never be "fixed", since
  the only way to do that is to edit a recording. Those surfaces are checked for
  the one absolute that would be a lie about the system itself.
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

AUTHORED_SUFFIXES = {".md", ".py", ".toml", ".yml", ".yaml"}
EVIDENCE_DIRS = {"results"}


def _authored_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    out = []
    for name in listed:
        path = ROOT / name
        if path.suffix not in AUTHORED_SUFFIXES or not path.exists():
            continue
        if path.parts[len(ROOT.parts)] in EVIDENCE_DIRS and path.suffix != ".md":
            continue
        out.append(path)
    return out


DISCLAIMERS = ("never", "not ", "avoid", "does not", "must not", "forbidden", "denied")


def _disclaimed(text: str, term: str) -> bool:
    """Does every line mentioning this term also disclaim it?"""
    lines = [ln for ln in text.splitlines() if re.search(rf"\b{re.escape(term)}\b", ln, re.I)]
    return bool(lines) and all(
        any(d in ln.lower() for d in DISCLAIMERS) for ln in lines
    )


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

        The sweep asserts it actually swept something. An earlier version globbed
        `demo/`, a directory that does not exist, so half of this test had never
        checked a byte — the same shape as every other defect this repo has
        caught: a check that cannot fail.
        """
        swept = 0
        for directory in EVIDENCE_DIRS:
            root = ROOT / directory
            assert root.is_dir(), (
                f"{directory}/ does not exist — this sweep would check nothing"
            )
            for path in root.glob("**/*"):
                if path.is_file() and path.suffix in {".log", ".txt", ".json"}:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    assert not _hits(text, DENIED_EVERYWHERE), path
                    swept += 1
        assert swept, "no captured evidence found to sweep"


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
