"""Invariant sweeps over AUTHORED PROSE.

The absolutes this repo claims about itself have to hold everywhere they could
be violated, not only where they are expected to. Two rules shape the sweep:

- Only prose the repo AUTHORS is swept: markdown, docstrings, commit subjects.
- Captured model output (`results/*.log`, `demo/`) is EVIDENCE, not authorship.
  A deny-listed word appearing in a genuine capture must never be "fixed", since
  the only way to do that is to edit a recording. Those surfaces are checked for
  the one absolute that would be a lie about the system itself.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Claims that would be untrue of this system, or that the research says read as
# naive to the audience this repo is written for.
DENIED = {
    "production-ready": "this is a demonstration system",
    "state-of-the-art": "no benchmark here supports it",
    "revolutionary": "marketing register",
    "replaces peer review": "explicitly out of scope",
    "autonomous reviewer": "explicitly out of scope",
    "world-class": "marketing register",
}

# The one absolute that must not appear even in a capture, because it would be a
# false claim about the system regardless of who wrote it.
DENIED_EVERYWHERE = {"production-ready"}

AUTHORED_SUFFIXES = {".md", ".py", ".toml", ".yml", ".yaml"}
EVIDENCE_DIRS = {"results", "demo"}


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
            hits = _hits(path.read_text(encoding="utf-8", errors="ignore"), DENIED)
            # LIMITATIONS and this test NAME the forbidden phrases in order to
            # disclaim them; that is the opposite of claiming them.
            if hits and path.name not in ("LIMITATIONS.md", "test_prose_invariants.py"):
                offenders[str(path.relative_to(ROOT))] = hits
        assert not offenders, offenders

    def test_no_denied_claims_in_commit_messages(self) -> None:
        log = subprocess.run(
            ["git", "log", "--format=%s"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
        assert not _hits(log, DENIED)

    def test_captured_evidence_is_never_edited_to_pass(self) -> None:
        """Captures are checked for the one absolute only — never groomed."""
        for directory in EVIDENCE_DIRS:
            for path in (ROOT / directory).glob("**/*"):
                if path.is_file() and path.suffix in {".log", ".txt"}:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    assert not _hits(text, DENIED_EVERYWHERE), path


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
