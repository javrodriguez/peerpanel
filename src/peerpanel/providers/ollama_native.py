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

from .base import WINDOW_SOURCE_NATIVE, ChatResponse
from .context import check_not_truncated, context_for_call, prompt_chars


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
        # The window is sized per call from the prompt and the output budget
        # (context.py): big enough that Ollama never cuts the prompt, small
        # enough that the KV cache stays on the GPU (a 32k window with parallel
        # slots spilled to CPU and crawled — DECISIONS.md D7). A fixed pin did
        # neither: 4,096 silently halved every reviewer and verifier prompt (D19).
        num_ctx = context_for_call(system, user, max_tokens, model=self._model)
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
                "num_ctx": num_ctx,
            },
        )
        prompt_tokens = resp.prompt_eval_count or 0
        check_not_truncated(
            prompt_tokens, prompt_chars(system, user), model=self._model, context=num_ctx
        )
        return ChatResponse(
            text=resp.message.content or "",
            model=self._model,
            prompt_tokens=prompt_tokens,
            completion_tokens=resp.eval_count or 0,
            context=num_ctx,
            # This wire SET the window it is reporting, in this call's own options —
            # the strongest form of the field, and a different one from the OpenAI
            # wire's read of the server's resident runner.
            context_source=WINDOW_SOURCE_NATIVE,
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
