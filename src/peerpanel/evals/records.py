"""The ONE rule a committed record's run conditions are read by, in one place.

Two test files ask the same question of the same records — `test_run_conditions.py`
sweeps every JSON under `results/`, `test_artifact_conformance.py` asks it of the
records it loads through the live schemas. The rule used to live inside the second
one, so the first could only re-implement it, and two implementations of one rule
drift until a record passes the sweep and fails the conformance test (or worse, the
other way round). It lives here now and both import it; the D11 twin-check shape.

It sits inside the package on purpose: `make lint` types and lints `src/`, so the
rule the published numbers depend on is held to the same bar as the code that
produced them. `assert` is deliberate — these are read from tests, and the failure
message is the point.

`derive_window_sources` is the emitter's own mapping, exposed as a pure function so
the two index-build records (which are NOT re-run — DECISIONS D20) can gain the
window-source field through exactly the derivation `graph build --publish` would
have written for the wire the record already names, and so a test can prove that is
what happened. Nothing else about a record is ever derived: a field the emitter
cannot produce, written into a record, is a mock.
"""

from __future__ import annotations

import copy
from typing import Any

from peerpanel.agents.schemas import CallStats
from peerpanel.providers.base import WINDOW_SOURCE_NATIVE, WINDOW_SOURCE_OPENAI

# The only strings a record may name as the source of the window its margin was
# measured against. A record naming anything else is naming a field nobody can read.
KNOWN_WINDOW_SOURCES = frozenset({WINDOW_SOURCE_NATIVE, WINDOW_SOURCE_OPENAI})

# The wire prefix of a provider name (`ollama-native:llama3.1:8b`) determines the
# window source by construction: the native wire SETS `options.num_ctx` per call, the
# OpenAI-compatible wire can only READ what the server loaded. This mapping is the
# emitter's — the ledger fills `window_sources` from the constant each wire stamps on
# its own responses, so a derivation through this table returns the same string.
_WIRE_WINDOW_SOURCE = {
    "ollama-native": WINDOW_SOURCE_NATIVE,
    "ollama-openai": WINDOW_SOURCE_OPENAI,
}

WINDOW_SOURCES_FIELD = "model_calls.window_sources"


def assert_run_conditions(
    stats: CallStats,
    where: str,
    *,
    replay_allowed: bool = False,
    record: dict[str, Any] | None = None,
) -> None:
    """The D19 proof: every prompt this record's calls sent fit the window it ran in.

    Read per call, not across the record: the largest prompt and the smallest window
    usually belong to different calls, so comparing those two refuses honest records
    (a 4,100-token reviewer prompt in an 8,192 window beside a short converger call in
    a 4,096 one) while proving nothing about either.

    `window_sources` is the second half of the same proof (requirement 8): the margin
    is a number, and this says which field on which wire it was measured against, so
    a reader can check the margin instead of trusting it.

    `replay_allowed` is for the index-build records only: `make graph` on a complete
    extraction cache replays every chunk and calls nothing. Such a record must PROVE
    it replayed, which is what `record` is for — without it there is nothing to check
    the claim against.
    """
    if stats.calls == 0:
        # A rebuild that replayed a complete cache called no model. It may not claim
        # measurements it never made — that would be a mock in the numbers.
        assert replay_allowed, f"{where}: a record with no model calls measured nothing"
        assert stats.largest_prompt_tokens == 0, f"{where}: no calls, but a prompt measured"
        assert stats.smallest_context is None and stats.smallest_headroom_tokens is None, where
        assert stats.window_sources == [], (
            f"{where}: no calls, but the record names {stats.window_sources} as the source "
            "of a window — nothing read a window"
        )
        assert record is not None, (
            f"{where}: a record with no model calls has to show it replayed a complete "
            "cache; this one was checked without the record that would show it"
        )
        # Nothing was extracted, so every chunk must already have been on disk. Only
        # this direction is sound: `cached_before` counts FILES in the cache directory
        # (pipeline.py) while a hit is keyed by chunk id, provider and PROMPT_VERSION,
        # so orphans and a bumped prompt version both inflate it above the number of
        # usable records — an honest cold build can show cached_before == chunks.
        assert record["cached_before"] >= record["chunks"], (
            f"{where}: no model calls, but only {record['cached_before']} cached "
            f"records for {record['chunks']} chunks"
        )
        return
    assert stats.largest_prompt_tokens > 0, where
    assert stats.smallest_context is not None, f"{where}: a wire that reports no window"
    assert stats.smallest_headroom_tokens is not None, f"{where}: no per-call margin recorded"
    assert stats.smallest_headroom_tokens > 0, (
        f"{where}: a prompt came within {stats.smallest_headroom_tokens} tokens of the window "
        f"it ran in (largest prompt {stats.largest_prompt_tokens:,}, smallest window "
        f"{stats.smallest_context:,}); a prompt was cut"
    )
    assert stats.window_sources, (
        f"{where}: {stats.calls} call(s) recorded a margin against a window whose source "
        "the record does not name — the margin cannot be checked (requirement 8)"
    )
    unknown = [s for s in stats.window_sources if s not in KNOWN_WINDOW_SOURCES]
    assert not unknown, (
        f"{where}: window source(s) no wire in this repository emits: {unknown}"
    )


def derive_window_sources(record: dict[str, Any]) -> dict[str, Any]:
    """The emitter's own derivation of `model_calls.window_sources`, as a pure function.

    The window source is a constant of the WIRE, and the record already names its wire
    in `provider`. So for a record this code emitted, `window_sources` is a function of
    fields the record carries — which is why the two index builds can gain it without
    being re-run, and why a test can prove the written value is the one the emitter
    would have written rather than a string someone chose.

    Returns a copy: the argument is never mutated, and a record that already carries
    the derived value comes back equal to itself (the idempotence the CLI door needs).
    Raises `ValueError` on a wire this repository has no window source for, or on a
    record that already names a DIFFERENT source — that is a real disagreement between
    the record and the wire it claims, and silently overwriting it would hide it.
    """
    provider = record.get("provider")
    if not isinstance(provider, str) or not provider:
        raise ValueError(
            f"cannot derive a window source: the record names no provider ({provider!r}); "
            "the derivation reads the wire from `provider` and nothing else"
        )
    wire = provider.split(":", 1)[0]
    source = _WIRE_WINDOW_SOURCE.get(wire)
    if source is None:
        raise ValueError(
            f"unknown wire {wire!r} (provider {provider!r}): this repository emits a window "
            f"source for {sorted(_WIRE_WINDOW_SOURCE)} only. A wire whose window source is "
            "not known cannot have one derived — re-run the record instead."
        )
    stats = record.get("model_calls")
    if not isinstance(stats, dict):
        raise ValueError(f"{provider}: the record carries no model_calls object to derive into")
    calls = stats.get("calls")
    if not isinstance(calls, int):
        raise ValueError(f"{provider}: model_calls.calls is {calls!r}, not a call count")
    # [] when the run called nothing: no call read a window, so no source was read.
    derived = [] if calls == 0 else [source]
    existing = stats.get("window_sources")
    if existing is not None and list(existing) != derived:
        raise ValueError(
            f"{provider}: the record already names {list(existing)} as its window source(s) "
            f"while its wire derives {derived}. A record and the wire it names disagree; "
            "that is a finding, not something to overwrite."
        )
    out = copy.deepcopy(record)
    out_stats = out["model_calls"]
    assert isinstance(out_stats, dict)  # narrowed above, on the same shape
    out_stats["window_sources"] = derived
    return out


def find_model_calls(tree: Any) -> list[tuple[str, dict[str, Any]]]:
    """Every `model_calls` object anywhere in a loaded record, with the path to it.

    The sweep cannot know where a record keeps its call stats — the build records hold
    one at the top level, a planted-error record holds one per arm under `results[i]` —
    and a rule that only looks where it expects is a rule that misses the layer nobody
    thought about, which is exactly how the community reports shipped with no record at
    all. So it walks, and the path comes back with the object so a failure names the
    arm it belongs to.
    """
    found: list[tuple[str, dict[str, Any]]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child = f"{path}.{key}" if path else str(key)
                if key == "model_calls" and isinstance(value, dict):
                    found.append((child, value))
                    continue
                walk(value, child)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")

    walk(tree, "")
    return found
