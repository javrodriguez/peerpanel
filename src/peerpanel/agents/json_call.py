"""One structured model call, retried once when the output was cut.

Every JSON-producing call in the panel and the baseline goes through here so
that the same two rules hold everywhere: temperature is what the caller asked,
and an unparseable answer earns exactly ONE retry at double the output budget.
An output that does not parse at 1,024 tokens is almost always an output that
ran out of room (constrained decoding closes the JSON only if it gets there);
doubling the budget is the fix, and both calls are ledgered so the retry is
paid for in every published token count.

The prompt itself is never trimmed here. A prompt that does not fit its window
is refused by the provider (providers/context.py) before the model sees it.
"""

from __future__ import annotations

import json
from typing import Any

from peerpanel.providers.base import ChatProvider, TokenLedger


def parse_object(text: str) -> dict[str, Any] | None:
    """A JSON object from the model's text, or None (a bare list is not an object)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def call_json(
    provider: ChatProvider,
    ledger: TokenLedger | None,
    *,
    system: str,
    user: str,
    schema: dict[str, Any],
    max_tokens: int,
    temperature: float = 0.0,
) -> dict[str, Any] | None:
    """One call, ONE retry at double the budget if the answer did not parse.

    Returns the parsed object, or None when both attempts failed to parse — the
    caller records that honestly (a `truncated` reviewer, a skipped sample),
    never as an empty success.
    """
    for budget in (max_tokens, max_tokens * 2):
        response = provider.chat(
            system=system,
            user=user,
            json_schema=schema,
            temperature=temperature,
            max_tokens=budget,
        )
        if ledger is not None:
            ledger.record(response)
        data = parse_object(response.text)
        if data is not None:
            return data
    return None
