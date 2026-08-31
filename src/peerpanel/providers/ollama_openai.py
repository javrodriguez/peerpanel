"""Ollama through the OpenAI-compatible wire protocol (the `openai` client).

One of the two locally-exercised chat wires (the other is the native Ollama
client in ollama_native.py — a genuinely different protocol, which is what
makes the seam claim real rather than model-swapping).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from openai import OpenAI

from .base import ChatResponse

DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaOpenAIChat:
    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL) -> None:
        # Ollama ignores the API key; the client requires a non-empty string.
        self._client = OpenAI(base_url=base_url, api_key="ollama-local")
        self._model = model

    @property
    def name(self) -> str:
        return f"ollama-openai:{self._model}"

    def chat(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse:
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
        return ChatResponse(
            text=resp.choices[0].message.content or "",
            model=self._model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
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
