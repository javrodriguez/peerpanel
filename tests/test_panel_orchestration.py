"""Exclusion inside `run_panel` — the half the ablation tests did not cover.

`test_exclusion_orchestration.py` closed this gap for `run_ablation` and named
the panel's identical guards as part of the same defect, but nothing imported
`run_panel`. That was demonstrated, not theorised: dropping `exclude_docs`,
neutering `build_run_graph`'s exclusion and deleting the comparability guard —
five mutations across `panel.py` and `planted_eval.py` — left the suite at 263
passed, byte-identical to clean, while a manuscript could retrieve its own
published twin.

These tests drive the REAL `run_panel` on the committed CI corpus with stub
models (no Ollama, no network) and fail when any of those guards is weakened.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from peerpanel.agents.schemas import PanelReview
from peerpanel.orchestration import PanelProviders, run_panel
from peerpanel.orchestration import panel as panel_mod
from peerpanel.providers.base import ChatResponse

ROOT = Path(__file__).resolve().parents[1]
TWIN = "PMC10729969"
MANUSCRIPT = ROOT / "manuscripts" / "met17-auxotroph.txt"


class _StubChat:
    """A model that answers everything with schema-shaped nothing.

    The point of these tests is the retrieval boundary, not model quality: the
    stub keeps them deterministic and fast while the REAL orchestration —
    index build, per-run graph rebuild, every guard — executes unchanged.
    """

    name = "stub"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, *, system: str, user: str, json_schema: object = None,
             temperature: float = 0.0, max_tokens: int = 2048) -> ChatResponse:
        self.prompts.append(user)
        if json_schema and "scores" in json.dumps(json_schema):
            body = ('{"scores": {"soundness": 3, "presentation": 3, "contribution": 3},'
                    ' "confidence": 3, "findings": []}')
        elif json_schema and "claims" in json.dumps(json_schema):
            body = '{"claims": []}'
        elif json_schema and "verdict" in json.dumps(json_schema):
            body = '{"verdict": "NOT_ENOUGH_INFO", "evidence": []}'
        else:
            body = "no synthesis"
        return ChatResponse(text=body, model="stub", prompt_tokens=1, completion_tokens=1)


def _providers() -> tuple[PanelProviders, _StubChat]:
    stub = _StubChat()
    return PanelProviders(methods=stub, novelty=stub, verifier=stub, converger=stub), stub


def _run() -> tuple[PanelReview, _StubChat]:
    providers, stub = _providers()
    return run_panel(ROOT, MANUSCRIPT, providers, corpus="ci"), stub


class TestPanelExclusion:
    def test_the_twin_is_withheld_and_the_run_says_so(self) -> None:
        review, _ = _run()
        assert review.excluded_docs == [TWIN]
        assert review.dropped_chunks > 0, (
            "a non-empty exclusion LIST that removed no chunks is the hollow state"
        )

    def test_no_twin_content_reaches_any_reviewer_prompt(self) -> None:
        """The end that matters: not what the index says, but what the model saw."""
        _, stub = _run()
        assert stub.prompts, "the panel made no model calls — the test proves nothing"
        offenders = [p for p in stub.prompts if f"{TWIN}:" in p]
        assert not offenders, f"{len(offenders)} prompts carried twin chunks"

    def test_the_guard_fires_when_exclusion_becomes_a_no_op(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reproduces the historical mutation: an index that excludes nothing."""
        real_build = panel_mod.Index.build

        def blind_build(chunks, vectors=None, exclude_docs=frozenset()):  # type: ignore[no-untyped-def]
            return real_build(chunks, vectors, exclude_docs=frozenset())

        monkeypatch.setattr(panel_mod.Index, "build", staticmethod(blind_build))
        providers, _ = _providers()
        with pytest.raises(RuntimeError, match="removed no chunks"):
            run_panel(ROOT, MANUSCRIPT, providers, corpus="ci")

    def test_the_graph_is_rebuilt_with_the_twin_withheld(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The per-run rebuild must be called with a NON-EMPTY exclusion set —
        passing set() would silently restore twin-derived entities and edges."""
        seen: list[set[str]] = []
        real = panel_mod.build_run_graph

        def spy(root, corpus, excluded):  # type: ignore[no-untyped-def]
            seen.append(set(excluded))
            return real(root, corpus, excluded)

        monkeypatch.setattr(panel_mod, "build_run_graph", spy)
        _run()
        assert seen, "run_panel never rebuilt the graph for this run"
        assert all(s == {TWIN} for s in seen), seen

    def test_an_unrecorded_manuscript_refuses_rather_than_guessing(
        self, tmp_path: Path
    ) -> None:
        """A manuscript with no twin-table row cannot have a sound exclusion set,
        so the run must refuse — never fall through to an unfiltered index."""
        header = MANUSCRIPT.read_text().split("# --- end attribution ---", 1)[0]
        rogue = tmp_path / "rogue.txt"
        rogue.write_text(
            header.replace("10.1101/2023.05.18.541364", "10.1101/9999.99.99.999999")
            + "# --- end attribution ---\nBody text.\n"
        )
        providers, _ = _providers()
        with pytest.raises(KeyError, match="not in the twin table"):
            run_panel(ROOT, rogue, providers, corpus="ci")
