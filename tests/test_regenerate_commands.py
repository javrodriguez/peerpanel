"""Every documented regenerate command must name a target that exists and writes
the artifact it is cited against.

Three separate defects in this repo were of this shape: a manifest naming
`make embeddings` for a fixture that target cannot write, and two Make targets
cited in RESULTS.md as producing logs they never tee'd. A reproduction
instruction that does not reproduce is worse than none — it converts a reader's
attempt to verify into a reason to distrust everything else.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (ROOT / "Makefile").read_text()


def _targets() -> set[str]:
    return set(re.findall(r"^([a-z][a-z0-9-]*):", MAKEFILE, re.MULTILINE))


def _recipe(target: str) -> str:
    match = re.search(rf"^{re.escape(target)}:\n((?:\t.*\n)+)", MAKEFILE, re.MULTILINE)
    return match.group(1) if match else ""


class TestManifestsNameRealTargets:
    def test_every_fixture_manifest_regenerates_via_a_real_target(self) -> None:
        manifests = sorted((ROOT / "fixtures").glob("*.manifest.json"))
        assert manifests, "no fixture manifests found"
        for path in manifests:
            command = json.loads(path.read_text())["regenerate"]
            target = command.removeprefix("make ").strip()
            assert target in _targets(), f"{path.name} names '{command}', not a Make target"

    def test_the_named_target_actually_writes_that_fixture(self) -> None:
        """The defect this catches: the demo manifest named the CI-only target."""
        for path in sorted((ROOT / "fixtures").glob("*.manifest.json")):
            target = json.loads(path.read_text())["regenerate"].removeprefix("make ").strip()
            recipe = _recipe(target)
            corpus = "demo" if "demo" in path.name else "ci"
            assert corpus in recipe or corpus == "ci", (
                f"{path.name} points at '{target}', whose recipe never mentions {corpus}"
            )


class TestResultsRegenerateColumn:
    def test_every_cited_command_exists(self) -> None:
        results = (ROOT / "results" / "RESULTS.md").read_text()
        cited = set(re.findall(r"`make ([a-z0-9-]+)`", results))
        assert cited, "RESULTS.md cites no regenerate commands"
        missing = sorted(c for c in cited if c not in _targets())
        assert not missing, f"RESULTS.md cites non-existent targets: {missing}"

    def test_targets_cited_for_logs_actually_write_them(self) -> None:
        """A target cited as producing a .log must tee to that path."""
        results = (ROOT / "results" / "RESULTS.md").read_text()
        pattern = r"\| `([a-z0-9-]+\.log)` \|[^|]+\| `make ([a-z0-9-]+)` \|"
        for log, target in re.findall(pattern, results):
            recipe = _recipe(target)
            assert log in recipe, (
                f"RESULTS.md cites `make {target}` as producing {log}, "
                f"but its recipe never writes that file"
            )
