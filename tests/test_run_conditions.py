"""Requirement 8, swept: every JSON under results/ says how its numbers were produced.

Round 3 found the community-report layer — the only input to the retrieval rung the
repository reports as its winner — shipping no run record at all, while the records
that DID carry run conditions were checked by one test that only looked at the records
it happened to load. Both halves are the same defect: a rule applied where someone
remembered to apply it.

So this file sweeps. Every `results/**/*.json` is either a model-run record, in which
case every `model_calls` object anywhere inside it passes the ONE rule in
`peerpanel.evals.records` (the same rule `test_artifact_conformance.py` imports — they
cannot drift), or it is named in `results/non-model-records.json` with the reason no
model produced it, in which case it must carry no run conditions at all. There is no
third state, and the manifest lists itself so no file sits outside the sweep.

A field written by derivation rather than measurement is disclosed the same way, in
`results/derived-fields.json`, and proven here to equal what the emitter itself would
have written for the wire the record names — never a string somebody chose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from peerpanel.agents.schemas import CallStats
from peerpanel.evals.planted_eval import PANEL_SYSTEM
from peerpanel.evals.records import (
    WINDOW_SOURCES_FIELD,
    assert_run_conditions,
    derive_window_sources,
    find_model_calls,
)
from peerpanel.providers.base import WINDOW_SOURCE_NATIVE, WINDOW_SOURCE_OPENAI

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

NON_MODEL_MANIFEST = "non-model-records.json"
DERIVED_MANIFEST = "derived-fields.json"
EVALUATION_LOOP = "evaluation-loop.json"

# The pinned evaluator prompt rounds 1-8 are convened under (the goal file's loop
# section). The README's evaluation-loop line quotes its first eight characters, so the
# published summary has to carry the whole thing and be held to it here.
EVALUATOR_SHA256 = "6e1bcda63fe82901ec75e0d465f102a86f67ccaee47f1de70f567da7e1551cd0"

# The sweep is by glob so a record lands inside it the moment it is written.
RESULT_JSON = sorted(p.relative_to(RESULTS).as_posix() for p in RESULTS.glob("**/*.json"))
BUILD_RECORDS = [n for n in RESULT_JSON if n.startswith("build-stats-")]
PANEL_RECORDS = [n for n in RESULT_JSON if n.startswith("panel-review-")]
PLANTED_RECORDS = [n for n in RESULT_JSON if n.startswith("planted-eval-")]

# A replay is legitimate for exactly one record class: `make graph` on a complete
# extraction cache calls nothing. Nothing else may report zero calls.
REPLAY_ALLOWED = tuple(BUILD_RECORDS)


def _read(name: str) -> dict[str, Any]:
    loaded = json.loads((RESULTS / name).read_text())
    assert isinstance(loaded, dict), f"{name}: a record is a JSON object"
    return loaded


def _read_or_say_why_it_is_required(name: str, why: str) -> dict[str, Any]:
    """Read a record that CP2/CP3 writes, or fail naming what depends on it.

    A missing file raises FileNotFoundError, which tells a reader the path and nothing
    about why the path has to be there. Every record this sweep requires is required
    for a reason, and the reason is what makes the red actionable.
    """
    assert (RESULTS / name).exists(), f"results/{name} is missing: {why}"
    return _read(name)


def _read_without_duplicate_keys(name: str) -> dict[str, Any]:
    """Load a manifest, refusing a key listed twice.

    `json.loads` keeps the last of two identical keys and says nothing, so a manifest
    that named a record twice — once exempting it, once with a different reason — would
    read as a clean single entry. The manifests are hand-maintained; this is the one
    way that hand can be checked.
    """
    path = RESULTS / name

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: dict[str, Any] = {}
        for key, value in items:
            if key in seen:
                raise AssertionError(f"{name}: {key!r} is listed twice")
            seen[key] = value
        return seen

    loaded = json.loads(path.read_text(), object_pairs_hook=pairs)
    assert isinstance(loaded, dict), f"{name}: a manifest is a JSON object"
    return loaded


def _manifest_records(name: str) -> dict[str, Any]:
    manifest = _read_without_duplicate_keys(name)
    assert manifest.get("_rule"), f"{name}: a manifest that does not state its rule exempts by fiat"
    entries = manifest["records"]
    assert isinstance(entries, dict), f"{name}: `records` maps a filename to why it is listed"
    return entries


def _call_stats(where: str, raw: dict[str, Any]) -> CallStats:
    """Load one `model_calls` object, or fail naming what the record cannot show."""
    try:
        return CallStats.model_validate(raw)
    except ValidationError as invalid:
        missing = sorted({str(e["loc"][0]) for e in invalid.errors() if e["type"] == "missing"})
        detail = f" — missing {missing}" if missing else f" — {invalid.errors()}"
        pytest.fail(
            f"{where}: this record's model_calls will not load under the current CallStats"
            f"{detail}. A field whose absence changes a number's meaning has no default "
            "(DECISIONS D18/D19): the record is re-run, never patched."
        )


class TestEveryResultsJsonSaysHowItWasProduced:
    """Item 1: the sweep. Two states, no third, and the manifest is in evaluators' scope."""

    @pytest.mark.parametrize("name", RESULT_JSON)
    def test_a_record_carries_its_run_conditions_or_is_a_named_exemption(self, name: str) -> None:
        doc = _read(name)
        exempt = _manifest_records(NON_MODEL_MANIFEST)
        found = find_model_calls(doc)
        if name in exempt:
            assert not found, (
                f"{name} is listed in results/{NON_MODEL_MANIFEST} as produced by no model, "
                f"yet it carries run conditions at {[path for path, _ in found]}. One of the "
                "two is wrong."
            )
            return
        assert found, (
            f"{name}: no model_calls anywhere in it, and it is not named in "
            f"results/{NON_MODEL_MANIFEST}. Every JSON under results/ either carries the run "
            "conditions of the model run that produced it or says why no model produced it "
            "(requirement 8)."
        )
        replay = name in REPLAY_ALLOWED
        for path, raw in found:
            where = f"{name}:{path}"
            assert_run_conditions(
                _call_stats(where, raw),
                where,
                replay_allowed=replay,
                record=doc if replay else None,
            )

    def test_the_sweep_saw_the_records_it_is_meant_to_sweep(self) -> None:
        """Item 7: non-vacuity. A glob over a directory that moved would pass silently."""
        assert len(RESULT_JSON) >= 10, (
            f"the sweep found only {len(RESULT_JSON)} JSON file(s) under {RESULTS}: "
            f"{RESULT_JSON}. This repository publishes index builds, community reports, "
            "panel runs, both evaluation arms, both ladders and its manifests — a sweep this "
            "small is not sweeping."
        )


class TestTheManifestsAreHonest:
    """Item 3: a manifest that names a file that is not there exempts nothing."""

    @pytest.mark.parametrize("manifest", [NON_MODEL_MANIFEST, DERIVED_MANIFEST])
    def test_it_names_only_files_that_exist(self, manifest: str) -> None:
        missing = [n for n in _manifest_records(manifest) if not (RESULTS / n).exists()]
        assert not missing, (
            f"results/{manifest} names {missing}, which do not exist under results/. An "
            "exemption for a file that is not there is an exemption nobody can check."
        )

    @pytest.mark.parametrize("manifest", [NON_MODEL_MANIFEST, DERIVED_MANIFEST])
    def test_it_names_nothing_twice(self, manifest: str) -> None:
        _read_without_duplicate_keys(manifest)

    @pytest.mark.parametrize("manifest", [NON_MODEL_MANIFEST, DERIVED_MANIFEST])
    def test_every_entry_says_why(self, manifest: str) -> None:
        empty = [n for n, why in _manifest_records(manifest).items() if not why]
        assert not empty, f"results/{manifest}: {empty} listed with no reason"

    def test_the_non_model_manifest_lists_itself(self) -> None:
        """It is a results JSON no model produced, so by its own rule it belongs on it."""
        entries = _manifest_records(NON_MODEL_MANIFEST)
        assert NON_MODEL_MANIFEST in entries, (
            f"results/{NON_MODEL_MANIFEST} exempts other files while sitting outside its own "
            "sweep; the one file the rule cannot reach is the file that states it."
        )
        assert DERIVED_MANIFEST in entries, (
            f"results/{DERIVED_MANIFEST} is a hand-maintained disclosure, not a model run"
        )


class TestDerivedFieldsEqualTheEmittersOwnDerivation:
    """Item 2: a derived field is the emitter's output for the wire the record names.

    The two index builds are not re-run (DECISIONS D20): re-extracting 1,077 chunks
    non-deterministically to write a string the wire determines would move every
    downstream number. The field they gain instead is derived by the same mapping the
    emitter uses, which is only defensible if a test proves it — twice over: the written
    value equals the derivation, and deriving again writes nothing new.
    """

    def _entries(self) -> dict[str, Any]:
        return _manifest_records(DERIVED_MANIFEST)

    def test_each_written_value_is_what_the_emitter_would_have_written(self) -> None:
        for name in sorted(self._entries()):
            record = _read(name)
            derived = derive_window_sources(record)
            assert record["model_calls"].get("window_sources") == (
                derived["model_calls"]["window_sources"]
            ), (
                f"{name}: the record carries {record['model_calls'].get('window_sources')!r} "
                f"while its own wire ({record.get('provider')!r}) derives "
                f"{derived['model_calls']['window_sources']!r}"
            )

    def test_deriving_again_changes_nothing(self) -> None:
        for name in sorted(self._entries()):
            record = _read(name)
            assert derive_window_sources(record) == record, (
                f"{name}: re-deriving rewrites the record, so the door is not idempotent"
            )

    def test_each_entry_says_what_was_derived_and_how_to_reproduce_it(self) -> None:
        for name, entry in sorted(self._entries().items()):
            assert isinstance(entry, dict), f"{DERIVED_MANIFEST}[{name}]: expected the D20 shape"
            assert entry.get("field") == WINDOW_SOURCES_FIELD, (
                f"{DERIVED_MANIFEST}[{name}]: names {entry.get('field')!r}; the only derived "
                f"field in this repository is {WINDOW_SOURCES_FIELD}"
            )
            assert entry.get("derived_from") == "provider", (
                f"{DERIVED_MANIFEST}[{name}]: the derivation reads the wire from `provider`"
            )
            assert entry.get("re_measured") is False, (
                f"{DERIVED_MANIFEST}[{name}]: a derived field was not measured, and the "
                "disclosure has to say so"
            )
            assert entry.get("regenerate"), (
                f"{DERIVED_MANIFEST}[{name}]: a disclosure without the command that would "
                "measure it instead leaves a reader nothing to do"
            )

    @pytest.mark.parametrize("name", BUILD_RECORDS)
    def test_a_record_not_disclosed_as_derived_carries_its_window_source_natively(
        self, name: str
    ) -> None:
        if name in self._entries():
            return
        stats = _read(name)["model_calls"]
        assert "window_sources" in stats, (
            f"{name}: no window source, and no entry in results/{DERIVED_MANIFEST} saying it "
            "was derived — the margin it publishes cannot be checked"
        )


_WHY_SUMMARIES = (
    "the community reports are the only input to graphrag-global's ranking and they ride "
    "the one wire that cannot set its own window, so their run conditions are a "
    "requirement, not a note (requirement 8). Regenerate cold: empty "
    "fixtures/summaries/<corpus>, then `make summaries` / `make demo-summaries`."
)

_WHY_LOOP = (
    "the first screen's evaluation-loop line reads its round count, its findings total and "
    "the pinned evaluator prompt's sha256 from it (requirement 10)"
)


class TestTheCommunityReportLayerPublishesItsRunConditions:
    """Item 4: the layer round 3 found shipping no record at all (report-2, finding 3b)."""

    @pytest.mark.parametrize("corpus", ["ci", "demo"])
    def test_the_record_exists_and_is_a_cold_run(self, corpus: str) -> None:
        name = f"summaries-stats-{corpus}.json"
        record = _read_or_say_why_it_is_required(name, _WHY_SUMMARIES)
        assert record["corpus"] == corpus, name
        assert record["provider"], f"{name}: the wire and model that served the reports"
        assert record["reports"] > 0, f"{name}: a record of no reports proves nothing"
        where = f"{name}:model_calls"
        stats = _call_stats(where, record["model_calls"])
        assert stats.calls >= record["reports"], (
            f"{name}: {stats.calls} call(s) for {record['reports']} report(s) — a report "
            "served from the committed cache makes no call, so this is a warm run and the "
            "committed record must be a cold one"
        )
        assert_run_conditions(stats, where)
        assert stats.window_sources == derive_window_sources(record)["model_calls"][
            "window_sources"
        ], (
            f"{name}: names {stats.window_sources} as its window source while its provider "
            f"({record['provider']}) is a different wire"
        )

    @pytest.mark.parametrize("corpus", ["ci", "demo"])
    def test_the_resolutions_are_the_ones_the_regenerate_command_asks_for(
        self, corpus: str
    ) -> None:
        """The record has to describe the run the documented command produces.

        `make summaries` asks for every Leiden level; `make demo-summaries` asks for the
        one level retrieval reads (DECISIONS D9). A record naming other levels was
        produced by something other than the command RESULTS.md cites.
        """
        record = _read_or_say_why_it_is_required(
            f"summaries-stats-{corpus}.json", _WHY_SUMMARIES
        )
        assert record["resolutions"] == _expected_resolutions(corpus), (
            f"summaries-stats-{corpus}.json reports resolutions {record['resolutions']}; the "
            f"Makefile recipe for this corpus produces {_expected_resolutions(corpus)}"
        )


def _summaries_recipes() -> dict[str, str]:
    """The `graph summaries` recipe line per corpus, read from the Makefile itself."""
    lines = re.findall(
        r"^\t.*python -m peerpanel graph summaries.*$", (ROOT / "Makefile").read_text(), re.M
    )
    assert lines, "no `graph summaries` recipe in the Makefile — this check reads nothing"
    recipes: dict[str, str] = {}
    for line in lines:
        corpus = re.search(r"--corpus\s+(\S+)", line)
        recipes[corpus.group(1) if corpus else "ci"] = line
    return recipes


def _expected_resolutions(corpus: str) -> list[str]:
    recipe = _summaries_recipes()[corpus]
    asked = re.search(r"--resolution\s+(\S+)", recipe)
    if asked:
        return [asked.group(1)]
    # No --resolution: the door summarises every level the build detected, and the
    # committed build record is where those levels are published.
    levels = _read(f"build-stats-{corpus}.json")["communities_per_resolution"]
    return sorted(levels, key=float)


class TestEveryModelLayerNamesTheModelThatServedIt:
    """Item 5: a run condition a reader can only learn from prose is not in the record."""

    @pytest.mark.parametrize("name", PANEL_RECORDS)
    def test_a_panel_record_names_every_role(self, name: str) -> None:
        record = _read(name)
        models = record.get("models")
        assert isinstance(models, dict) and models, (
            f"{name}: the reviewers were named per output and the verifier and converger "
            "were named nowhere a reader could check"
        )
        expected = {o["reviewer"] for o in record["reviewer_outputs"]} | {"verifier", "converger"}
        assert set(models) == expected, f"{name}: names {sorted(models)}, ran {sorted(expected)}"
        assert len(models) == 4, (
            f"{name}: PanelProviders serves four roles (methods, novelty, verifier, "
            f"converger); this record names {len(models)}"
        )
        assert all(models.values()), f"{name}: a role named with no model"

    @pytest.mark.parametrize("name", PLANTED_RECORDS)
    def test_each_evaluation_arm_names_the_model_that_served_it(self, name: str) -> None:
        for arm in _read(name)["results"]:
            models = arm.get("models")
            where = f"{name}:{arm['system']}"
            assert isinstance(models, dict) and models, f"{where}: no model named"
            assert all(models.values()), f"{where}: a role named with no model"
            if arm["system"] == PANEL_SYSTEM:
                assert len(models) == 4, f"{where}: the panel arm names {len(models)} roles"
                continue
            assert set(models) == {"single-agent"}, (
                f"{where}: a single-agent arm serves one role; this names {sorted(models)}"
            )


class TestTheEvaluationLoopSummary:
    """Item 6: the numbers the README's evaluation-loop line reads, and the private record.

    Requirement 10 lets a sentence about what the loop found say the record is private
    instead of quoting it. That branch is only honest if the published counts are held
    to something, so they live in a record of their own and the pin is checked here.
    """

    def test_it_exists(self) -> None:
        _read_or_say_why_it_is_required(EVALUATION_LOOP, _WHY_LOOP)

    def test_it_carries_the_counts_and_the_pin(self) -> None:
        record = _read_or_say_why_it_is_required(EVALUATION_LOOP, _WHY_LOOP)
        rounds = record["rounds_completed"]
        by_round = record["findings_by_round"]
        assert isinstance(rounds, int) and rounds > 0, f"{EVALUATION_LOOP}: {rounds!r} rounds"
        assert isinstance(by_round, list) and all(isinstance(n, int) for n in by_round), (
            f"{EVALUATION_LOOP}: findings_by_round is {by_round!r}, not a count per round"
        )
        assert len(by_round) == rounds, (
            f"{EVALUATION_LOOP}: {rounds} rounds completed but {len(by_round)} findings "
            "counts — every round that ran filed a number, including zero"
        )
        assert record["evaluator_sha256"] == EVALUATOR_SHA256, (
            f"{EVALUATION_LOOP}: the loop is only a loop while every round is convened under "
            "the same prompt; this record names a different one"
        )
        assert record["record"] == "private", (
            f"{EVALUATION_LOOP}: the gauntlet's own record is a private nested repository, "
            "and the published summary says so rather than implying an open one"
        )


class TestTheOneRule:
    """`assert_run_conditions` is the rule two files ask of every record: unit-test it."""

    @staticmethod
    def _stats(**over: Any) -> CallStats:
        base: dict[str, Any] = {
            "calls": 3,
            "largest_prompt_tokens": 5238,
            "smallest_context": 8192,
            "smallest_headroom_tokens": 2747,
            "window_sources": [WINDOW_SOURCE_NATIVE],
        }
        return CallStats.model_validate({**base, **over})

    def test_a_measured_record_passes(self) -> None:
        assert_run_conditions(self._stats(), "unit")

    def test_a_prompt_that_did_not_fit_is_refused(self) -> None:
        with pytest.raises(AssertionError, match="a prompt was cut"):
            assert_run_conditions(self._stats(smallest_headroom_tokens=-4), "unit")

    def test_a_margin_with_no_named_source_is_refused(self) -> None:
        with pytest.raises(AssertionError, match="does not name"):
            assert_run_conditions(self._stats(window_sources=[]), "unit")

    def test_a_source_no_wire_emits_is_refused(self) -> None:
        with pytest.raises(AssertionError, match="no wire in this repository emits"):
            assert_run_conditions(self._stats(window_sources=["I read it somewhere"]), "unit")

    def test_a_record_with_no_calls_measured_nothing(self) -> None:
        empty = self._stats(
            calls=0,
            largest_prompt_tokens=0,
            smallest_context=None,
            smallest_headroom_tokens=None,
            window_sources=[],
        )
        with pytest.raises(AssertionError, match="measured nothing"):
            assert_run_conditions(empty, "unit")

    def test_a_replay_has_to_show_it_replayed(self) -> None:
        empty = self._stats(
            calls=0,
            largest_prompt_tokens=0,
            smallest_context=None,
            smallest_headroom_tokens=None,
            window_sources=[],
        )
        assert_run_conditions(
            empty, "unit", replay_allowed=True, record={"cached_before": 234, "chunks": 234}
        )
        with pytest.raises(AssertionError, match="cached"):
            assert_run_conditions(
                empty, "unit", replay_allowed=True, record={"cached_before": 3, "chunks": 234}
            )
        with pytest.raises(AssertionError, match="has to show it replayed"):
            assert_run_conditions(empty, "unit", replay_allowed=True)

    def test_a_replay_may_not_carry_a_window_source_it_never_read(self) -> None:
        with pytest.raises(AssertionError, match="nothing read a window"):
            assert_run_conditions(
                self._stats(
                    calls=0,
                    largest_prompt_tokens=0,
                    smallest_context=None,
                    smallest_headroom_tokens=None,
                ),
                "unit",
                replay_allowed=True,
                record={"cached_before": 234, "chunks": 234},
            )


class TestTheDerivation:
    """`derive_window_sources` is pure and wire-determined, or the field is a mock."""

    @staticmethod
    def _record(provider: str, calls: int = 234, **stats: Any) -> dict[str, Any]:
        return {
            "corpus": "ci",
            "provider": provider,
            "cached_before": 0,
            "chunks": 234,
            "model_calls": {
                "calls": calls,
                "largest_prompt_tokens": 5238 if calls else 0,
                "smallest_context": 4096 if calls else None,
                "smallest_headroom_tokens": 2747 if calls else None,
                **stats,
            },
        }

    def test_the_native_wire_derives_the_window_it_configured(self) -> None:
        out = derive_window_sources(self._record("ollama-native:llama3.1:8b"))
        assert out["model_calls"]["window_sources"] == [WINDOW_SOURCE_NATIVE]

    def test_the_openai_wire_derives_the_window_it_read(self) -> None:
        out = derive_window_sources(self._record("ollama-openai:llama3.1:8b"))
        assert out["model_calls"]["window_sources"] == [WINDOW_SOURCE_OPENAI]

    def test_a_run_that_called_nothing_read_no_window(self) -> None:
        out = derive_window_sources(self._record("ollama-native:llama3.1:8b", calls=0))
        assert out["model_calls"]["window_sources"] == []

    def test_it_never_touches_the_record_it_is_given(self) -> None:
        record = self._record("ollama-native:llama3.1:8b")
        before = json.dumps(record, sort_keys=True)
        derive_window_sources(record)
        assert json.dumps(record, sort_keys=True) == before

    def test_deriving_an_already_derived_record_returns_it_unchanged(self) -> None:
        once = derive_window_sources(self._record("ollama-native:llama3.1:8b"))
        assert derive_window_sources(once) == once

    def test_a_record_naming_a_different_source_is_a_disagreement_not_an_overwrite(self) -> None:
        record = self._record(
            "ollama-native:llama3.1:8b", window_sources=[WINDOW_SOURCE_OPENAI]
        )
        with pytest.raises(ValueError, match="disagree"):
            derive_window_sources(record)

    def test_an_unknown_wire_cannot_have_a_window_source_derived(self) -> None:
        with pytest.raises(ValueError, match="unknown wire"):
            derive_window_sources(self._record("anthropic:claude-3"))

    def test_a_record_with_no_provider_derives_nothing(self) -> None:
        record = self._record("ollama-native:llama3.1:8b")
        del record["provider"]
        with pytest.raises(ValueError, match="names no provider"):
            derive_window_sources(record)

    def test_a_record_with_no_call_stats_derives_nothing(self) -> None:
        record = self._record("ollama-native:llama3.1:8b")
        del record["model_calls"]
        with pytest.raises(ValueError, match="no model_calls"):
            derive_window_sources(record)


class TestFindingEveryLayer:
    """The walk is what makes the sweep a sweep: it looks where nobody remembered to."""

    def test_it_finds_a_top_level_record(self) -> None:
        assert find_model_calls({"model_calls": {"calls": 1}}) == [("model_calls", {"calls": 1})]

    def test_it_finds_one_per_arm_with_the_path_to_it(self) -> None:
        tree = {"results": [{"model_calls": {"calls": 1}}, {"model_calls": {"calls": 2}}]}
        assert find_model_calls(tree) == [
            ("results[0].model_calls", {"calls": 1}),
            ("results[1].model_calls", {"calls": 2}),
        ]

    def test_it_reaches_through_nested_lists_and_dicts(self) -> None:
        tree = {"a": [{"b": {"c": [{"model_calls": {"calls": 9}}]}}]}
        assert find_model_calls(tree) == [("a[0].b.c[0].model_calls", {"calls": 9})]

    def test_a_record_with_none_comes_back_empty(self) -> None:
        assert find_model_calls({"corpus": "ci", "results": [1, 2, "three"]}) == []

    def test_a_string_named_model_calls_is_not_a_record(self) -> None:
        """`derived-fields.json` names the field `model_calls.window_sources` as a VALUE;
        a manifest describing run conditions is not a record carrying them."""
        assert find_model_calls({"field": WINDOW_SOURCES_FIELD, "model_calls": "none"}) == []
