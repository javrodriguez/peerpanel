"""Provider seams: every model call in PeerPanel flows through these Protocols.

Two chat adapters exist over genuinely different wire protocols (the OpenAI
client pointed at Ollama, and the native Ollama client), plus an Anthropic
adapter whose send path is unexercised on this machine (see anthropic_wire).
Two embedding backends exist (Ollama nomic-embed-text, model2vec pure-CPU).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
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
    # The window the call ran in, when the wire knows it. The ledger keeps the
    # smallest seen so a record can state that every call fit (context.py).
    context: int | None = None


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

    The review panel and the budget-matched single-agent baseline are both
    driven through the same ledger type so their token spends are comparable
    by construction — a contract-drift test holds the two to it. Recording is
    locked: the verifier judges claims from a thread pool.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    largest_prompt_tokens: int = 0
    smallest_context: int | None = None
    # The margin between a prompt and the window IT ran in, smallest across the calls.
    # The two fields above cannot show this: the largest prompt and the smallest window
    # are usually different calls, so a record where a 4,100-token reviewer prompt ran in
    # an 8,192 window and a short converger call ran in a 4,096 one looks like a cut
    # prompt while every call was read whole. Positive here means every prompt fit.
    smallest_headroom_tokens: int | None = None
    # Reviewers and the verifier record from worker threads; three unguarded `+=`
    # would lose counts under contention, and a lost count is a wrong published number.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, response: ChatResponse) -> None:
        with self._lock:
            self.prompt_tokens += response.prompt_tokens
            self.completion_tokens += response.completion_tokens
            self.calls += 1
            self.largest_prompt_tokens = max(self.largest_prompt_tokens, response.prompt_tokens)
            if response.context is not None:
                self.smallest_context = (
                    response.context
                    if self.smallest_context is None
                    else min(self.smallest_context, response.context)
                )
                headroom = response.context - response.prompt_tokens
                self.smallest_headroom_tokens = (
                    headroom
                    if self.smallest_headroom_tokens is None
                    else min(self.smallest_headroom_tokens, headroom)
                )

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
