"""Ollama through the OpenAI-compatible wire protocol (the `openai` client).

One of the two locally-exercised chat wires (the other is the native Ollama
client in ollama_native.py — a genuinely different protocol, which is what
makes the seam claim real rather than model-swapping).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import ollama
from numpy.typing import NDArray
from openai import OpenAI

from .base import ChatResponse
from .context import assert_fits, check_not_truncated, prompt_chars

DEFAULT_BASE_URL = "http://localhost:11434/v1"

WINDOW_REMEDY = (
    "The OpenAI-compatible endpoint cannot set the window per call. Start Ollama with "
    "a larger one (OLLAMA_CONTEXT_LENGTH=16384; the desktop app reads it from "
    "`launchctl setenv` on macOS and needs a restart), or run this role on the native "
    "wire, which sizes the window per call."
)


class OllamaOpenAIChat:
    """The window is whatever the server loaded the model with (its default is
    4,096) and this wire cannot change it, so it is READ and every call is held
    to it before it is sent (context.py). A prompt that would be cut is refused
    with the remedy; nothing this wire returns was read in part."""

    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL) -> None:
        # Ollama ignores the API key; the client requires a non-empty string.
        self._client = OpenAI(base_url=base_url, api_key="ollama-local")
        self._native = ollama.Client(host=base_url.removesuffix("/v1").removesuffix("/"))
        self._model = model
        self._context: int | None = None

    @property
    def name(self) -> str:
        return f"ollama-openai:{self._model}"

    def _loaded_context(self) -> int | None:
        """The window the server has this model loaded in, if it is loaded."""
        for running in self._native.ps().models:
            if running.model == self._model and running.context_length:
                return int(running.context_length)
        return None

    def context(self) -> int:
        """The window THIS wire's calls run in, read once.

        A one-token call through this wire goes first, always: the server reloads
        the runner whenever a request's window differs from the resident one
        (measured on 0.33: a model left at 8,192 by the native wire comes back at
        the 4,096 default on the next OpenAI-wire call), so a `ps` read taken
        before this wire has spoken can report another wire's window. After the
        warm-up, `ps` reports the server default — the only window this wire ever
        gets, fixed for the server's lifetime."""
        if self._context is None:
            self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "ok"}],
                max_tokens=1,
            )
            found = self._loaded_context()
            if found is None:
                raise RuntimeError(
                    f"{self._model}: Ollama did not report a context window for the "
                    "loaded model (ps); this wire cannot hold prompts to a window it "
                    "cannot read."
                )
            self._context = found
        return self._context

    def chat(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        context = self.context()
        assert_fits(context, system, user, max_tokens, model=self._model, remedy=WINDOW_REMEDY)
        extra: dict[str, Any] = {}
        if json_schema is not None:
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "structured_output", "schema": json_schema},
            }
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
        usage = resp.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        check_not_truncated(
            prompt_tokens, prompt_chars(system, user), model=self._model, context=context
        )
        return ChatResponse(
            text=resp.choices[0].message.content or "",
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=usage.completion_tokens if usage else 0,
            context=context,
        )


class OllamaOpenAIEmbed:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = DEFAULT_BASE_URL,
        dim: int = 768,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key="ollama-local")
        self._model = model
        self._dim = dim

    @property
    def name(self) -> str:
        return f"ollama-openai-embed:{self._model}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        out = np.asarray([item.embedding for item in resp.data], dtype=np.float32)
        if out.shape != (len(texts), self._dim):
            raise ValueError(f"embedding shape {out.shape} != ({len(texts)}, {self._dim})")
        return out
