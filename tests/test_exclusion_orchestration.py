"""Exclusion at the layer where it actually happens: the orchestration.

Every exclusion guard in this project lives inside `run_ablation` (and its
sibling in the panel), and until this file existed no test imported either.
That gap was demonstrated, not theorised: disabling exclusion in five places —
the `exclude_docs` argument and the `dropped_chunk_count` guard in
`evals/ablation.py`, the same two in `orchestration/panel.py`, and the
ground-truth subtraction in `evals/retrieval_eval.py` — left the suite at
252 passed, byte-identical to clean. Every reported number would then have been
computed over an index containing the manuscript's own published twin, with
nothing anywhere saying so.

These tests drive the REAL `run_ablation` over the committed CI corpus and fail
if exclusion is weakened anywhere along that path — including the mutation
above, which `test_the_guard_actually_fires` reproduces directly.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.typing import NDArray

from peerpanel.embeddings import store
from peerpanel.evals import ablation as ablation_mod
from peerpanel.evals.ablation import run_ablation
from peerpanel.retrieval import Index

ROOT = Path(__file__).resolve().parents[1]
TWIN = "PMC10729969"
RUNGS = {"bm25", "vector", "rrf-hybrid", "graphrag-local", "graphrag-global"}


class _TwinVectorEmbed:
    """Asks every rung the most dangerous question: "return the excluded document".

    The vector returned is the twin's OWN committed chunk vector, read from the
    real embedding fixture — nothing is synthesised. Against an unexcluded index
    the vector rungs would therefore rank that chunk first by construction
    (cosine 1.0 with itself), so exclusion is being proven under its worst case
    rather than an average one.
    """

    def __init__(self, root: Path) -> None:
        ids, vectors = store.load(root / "fixtures" / "ci_embeddings.npz")
        position = next(i for i, cid in enumerate(ids) if cid.startswith(f"{TWIN}:"))
        self._vector = vectors[position].astype(np.float32)

    @property
    def name(self) -> str:
        return "twin-chunk-vector (committed fixture, no model call)"

    @property
    def dim(self) -> int:
        return int(self._vector.shape[0])

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        return np.repeat(self._vector[None, :], len(texts), axis=0)


@pytest.fixture(scope="module")
def embedder() -> _TwinVectorEmbed:
    return _TwinVectorEmbed(ROOT)


class TestRealOrchestration:
    def test_ablation_runs_over_committed_bytes(self, embedder: _TwinVectorEmbed) -> None:
        """The path exists and completes — which itself proves the per-case guard
        passed, since run_ablation raises when a case drops nothing."""
        report = run_ablation(ROOT, embedder, corpus="ci")
        assert report.n_cases >= 1
        assert report.results

    def test_every_rung_is_exercised(self, embedder: _TwinVectorEmbed) -> None:
        """Closes the 'every mode' gap: rrf-hybrid is swept here too, so a future
        change that fused an unfiltered source would be seen."""
        report = run_ablation(ROOT, embedder, corpus="ci")
        assert {r.rung for r in report.results} == RUNGS

    def test_no_excluded_document_reaches_any_ranking(
        self, embedder: _TwinVectorEmbed, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Observe the real orchestration's own output rather than recomputing it:
        every ranking run_ablation builds, for every case and every rung."""
        seen: list[list[str]] = []
        real = ablation_mod.doc_ranking

        def spy(hits: list) -> list[str]:  # type: ignore[type-arg]
            ranking = real(hits)
            seen.append(ranking)
            return ranking

        monkeypatch.setattr(ablation_mod, "doc_ranking", spy)
        report = run_ablation(ROOT, embedder, corpus="ci")

        assert len(seen) == report.n_cases * len(RUNGS)
        assert any(seen), "rankings were empty — the assertion below would be vacuous"
        for ranking in seen:
            assert TWIN not in ranking


class TestTheGuardIsLive:
    def test_the_guard_actually_fires(
        self, embedder: _TwinVectorEmbed, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reproduces the mutation that used to pass: strip `exclude_docs` at the
        call site and the run must refuse, loudly, naming the law."""

        def blind_build(chunks: list, vectors: object, **_: object) -> Index:  # type: ignore[type-arg]
            return Index.build(chunks, vectors)  # type: ignore[arg-type]

        monkeypatch.setattr(ablation_mod, "Index", SimpleNamespace(build=blind_build))
        with pytest.raises(RuntimeError, match="removed no chunks"):
            run_ablation(ROOT, embedder, corpus="ci")
