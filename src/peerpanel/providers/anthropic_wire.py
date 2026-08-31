"""Anthropic Messages API adapter.

UNEXERCISED ADAPTER — no provider call has ever been made from this repo: no
Anthropic API key exists on the development machine and none was requested.
What IS tested is the request construction (`build_request`), asserted
wire-shape-only in tests. The send path raises MissingAPIKeyError until an
ANTHROPIC_API_KEY is present in the caller's environment. There are no
recorded response fixtures for this adapter, deliberately: a fixture nobody
recorded would be a mock wearing a recording's label.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import ChatResponse

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
STRUCTURED_TOOL_NAME = "emit_structured_output"


class MissingAPIKeyError(RuntimeError):
    """Raised when the send path is reached without ANTHROPIC_API_KEY set."""


def build_request(
    *,
    model: str,
    system: str,
    user: str,
    json_schema: dict[str, Any] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> dict[str, Any]:
    """Construct the Messages API request body (pure; wire-shape tested)."""
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if json_schema is not None:
        # Structured output via a forced tool call carrying the schema.
        body["tools"] = [
            {
                "name": STRUCTURED_TOOL_NAME,
                "description": "Return the structured result.",
                "input_schema": json_schema,
            }
        ]
        body["tool_choice"] = {"type": "tool", "name": STRUCTURED_TOOL_NAME}
    return body


class AnthropicChat:
    def __init__(self, model: str = "claude-sonnet-5") -> None:
        self._model = model

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def chat(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise MissingAPIKeyError(
                "No ANTHROPIC_API_KEY in the environment. This adapter has never "
                "been exercised in this repo; set a key to make it live."
            )
        body = build_request(
            model=self._model,
            system=system,
            user=user,
            json_schema=json_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        resp = httpx.post(
            API_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
            json=body,
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
            elif block.get("type") == "tool_use" and json_schema is not None:
                import json as _json

                text = _json.dumps(block.get("input", {}))
        usage = data.get("usage", {})
        return ChatResponse(
            text=text,
            model=self._model,
            prompt_tokens=int(usage.get("input_tokens", 0)),
            completion_tokens=int(usage.get("output_tokens", 0)),
        )
