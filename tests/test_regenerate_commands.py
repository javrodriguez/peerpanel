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
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

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


ROW = re.compile(r"^\| (`[^|]+`(?: / `\.log`)?) \| [^|]+ \| `make ([^`]+)`[^|]*\|$", re.MULTILINE)


def _rows() -> list[tuple[list[str], str]]:
    """Every row of the RESULTS.md regenerate table: (files it names, make command)."""
    results = (ROOT / "results" / "RESULTS.md").read_text()
    rows: list[tuple[list[str], str]] = []
    for files_cell, command in ROW.findall(results):
        names = re.findall(r"`([^`]+)`", files_cell)
        files = [names[0]]
        for extra in names[1:]:  # "`x.json` / `.log`" means x.json and x.log
            files.append(Path(names[0]).with_suffix(extra).name if extra.startswith(".") else extra)
        rows.append((files, command.strip()))
    assert rows, "RESULTS.md has no regenerate table rows"
    return rows


def _dry_run(command: str) -> str:
    if shutil.which("make") is None:
        pytest.skip("make is not installed")
    proc = subprocess.run(
        ["make", "-n", *command.split()], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, f"`make -n {command}` failed:\n{proc.stderr}"
    return proc.stdout


def _where(dry_run: str) -> set[str]:
    """Run each `python -m peerpanel …` line of a dry run with --where; the paths
    the commands say they would write, relative to the repo root."""
    written: set[str] = set()
    for line in dry_run.splitlines():
        if "-m peerpanel" not in line:
            continue
        args = line.split("-m peerpanel", 1)[1].split("2>&1", 1)[0].split("|", 1)[0].split()
        proc = subprocess.run(
            [sys.executable, "-m", "peerpanel", *args, "--where"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"`peerpanel {' '.join(args)} --where` failed:\n{proc.stderr}"
        written.update(proc.stdout.split())
    return written


class TestPipedRecipesFailLoudly:
    """A target that pipes a real run through `tee` must not report success when the run
    crashed. Without `set -o pipefail` the exit code is tee's — always 0 — so the log is
    left empty or half-written and the stale record beside it stands as if regenerated.
    That is the empty-log-with-no-record shape, and it ships silently."""

    def test_every_teed_recipe_guards_its_pipe(self) -> None:
        makefile = (ROOT / "Makefile").read_text()
        assert "SHELL := /bin/bash" in makefile, "pipefail needs bash, not /bin/sh"
        unguarded = [
            line.strip()
            for line in makefile.splitlines()
            if line.startswith("\t") and "| tee " in line and "set -o pipefail" not in line
        ]
        assert not unguarded, f"piped recipes with no pipefail guard: {unguarded}"

    def test_the_guard_is_what_makes_the_difference(self) -> None:
        """GNU Make 3.81 — what macOS ships, and where these runs happen — ignores
        .SHELLFLAGS silently, so the guard has to live in the recipe.

        Both arms are run, because the guarded arm alone proves nothing: `make` returns
        non-zero for plenty of reasons, and an earlier version of this test passed
        identically with no SHELL line at all. The control arm is the whole test — the
        SAME recipe without the guard must report success, or the guard is not what is
        being measured.
        """
        if shutil.which("make") is None:
            pytest.skip("make is not installed")

        def run(guard: str) -> int:
            with tempfile.TemporaryDirectory() as tmp:
                Path(tmp, "Makefile").write_text(
                    f"SHELL := /bin/bash\n\nprobe:\n\t{guard}false | tee /dev/null\n"
                )
                return subprocess.run(
                    ["make", "probe"], cwd=tmp, capture_output=True, text=True, check=False
                ).returncode

        unguarded = run("")
        guarded = run("set -o pipefail; ")
        assert unguarded == 0, (
            "control arm did not reproduce the defect: a failed command piped into tee "
            f"returned {unguarded}, so this test is measuring something else"
        )
        assert guarded != 0, "the guard did not make a failed pipe fail"


class TestResultsRegenerateColumn:
    def test_every_cited_command_exists(self) -> None:
        results = (ROOT / "results" / "RESULTS.md").read_text()
        cited = set(re.findall(r"`make ([a-z0-9-]+)", results))
        assert cited, "RESULTS.md cites no regenerate commands"
        missing = sorted(c for c in cited if c not in _targets())
        assert not missing, f"RESULTS.md cites non-existent targets: {missing}"

    def test_the_table_names_every_committed_record_and_nothing_else(self) -> None:
        """A record the table does not name has no regenerate command; a row naming
        a file that is not committed describes evidence that does not exist."""
        named = {f for files, _ in _rows() for f in files}
        committed = {p.name for p in (ROOT / "results").iterdir() if p.suffix in {".json", ".log"}}
        assert named == committed, (
            f"unnamed records: {sorted(committed - named)}; "
            f"rows without a file: {sorted(named - committed)}"
        )

    def test_each_cited_command_writes_the_file_in_its_row(self) -> None:
        """The whole defect class, closed: expand the cited `make` line, ask the
        underlying command where it would write (--where, no model, no work), and
        require that to be the row's file. A log row must be tee'd to that path."""
        for files, command in _rows():
            dry_run = _dry_run(command)
            written = _where(dry_run)
            for name in files:
                if name.endswith(".log"):
                    assert f"tee results/{name}" in dry_run, (
                        f"`make {command}` is cited for {name} but never tees to it:\n{dry_run}"
                    )
                else:
                    assert f"results/{name}" in written, (
                        f"`make {command}` is cited for {name} but its command writes {written}"
                    )
