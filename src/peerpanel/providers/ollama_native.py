"""Ollama through its native client and wire protocol.

The second locally-exercised chat wire (see ollama_openai.py for the first).
The native protocol differs on the wire: structured outputs travel as
`format=<schema>`, token counts come back as prompt_eval_count/eval_count.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import ollama
from numpy.typing import NDArray

from .base import ChatResponse

# One serving slot's context. Prompts are budgeted to fit inside it.
NUM_CTX = 4096


class OllamaNativeChat:
    def __init__(self, model: str, host: str | None = None) -> None:
        self._client = ollama.Client(host=host)
        self._model = model

    @property
    def name(self) -> str:
        return f"ollama-native:{self._model}"

    def chat(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse:
        resp = self._client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format=json_schema,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                # Pin the context window. Without it the server uses the model's
                # own default (32k for qwen2), and with parallel slots that KV
                # cache spills to CPU and generation crawls — measured.
                "num_ctx": NUM_CTX,
            },
        )
        return ChatResponse(
            text=resp.message.content or "",
            model=self._model,
            prompt_tokens=resp.prompt_eval_count or 0,
            completion_tokens=resp.eval_count or 0,
        )


class OllamaNativeEmbed:
    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str | None = None,
        dim: int = 768,
    ) -> None:
        self._client = ollama.Client(host=host)
        self._model = model
        self._dim = dim

    @property
    def name(self) -> str:
        return f"ollama-native-embed:{self._model}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        resp = self._client.embed(model=self._model, input=texts)
        out = np.asarray(resp.embeddings, dtype=np.float32)
        if out.shape != (len(texts), self._dim):
            raise ValueError(f"embedding shape {out.shape} != ({len(texts)}, {self._dim})")
        return out
