"""The tier-1 quickstart must run the ENTIRE deterministic chain from committed
bytes — corpus, extraction cache, summary cache, embedding fixture — with a
provider that raises on any model call. This is CI's deepest integration test."""

from __future__ import annotations

import time
from pathlib import Path

from peerpanel.quickstart import run_quickstart

ROOT = Path(__file__).resolve().parents[1]


class TestQuickstart:
    def test_runs_from_committed_bytes_only_and_fast(self, capsys: object) -> None:
        t0 = time.monotonic()
        assert run_quickstart(ROOT) == 0
        assert time.monotonic() - t0 < 60  # the README's tier-1 promise
