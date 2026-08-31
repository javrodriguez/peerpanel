"""Provider seams: every model call in PeerPanel flows through these Protocols.

Two chat adapters exist over genuinely different wire protocols (the OpenAI
client pointed at Ollama, and the native Ollama client), plus an Anthropic
adapter whose send path is unexercised on this machine (see anthropic_wire).
Two embedding backends exist (Ollama nomic-embed-text, model2vec pure-CPU).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ChatResponse:
    """One completed chat call, with the token counts the ledger records."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


@runtime_checkable
class ChatProvider(Protocol):
    @property
    def name(self) -> str: ...

    def chat(
        self,
        *,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> ChatResponse: ...


@runtime_checkable
class EmbedProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed(self, texts: list[str]) -> NDArray[np.float32]: ...


@dataclass
class TokenLedger:
    """The ONE accounting counter.

    The review panel and the equal-compute single-agent baseline are both
    driven through the same ledger type so their token budgets are comparable
    by construction — a contract-drift test holds the two to it.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def record(self, response: ChatResponse) -> None:
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
