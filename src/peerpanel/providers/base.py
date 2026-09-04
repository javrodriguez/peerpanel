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

# Where a call's window came from, per wire — the string a record carries beside the
# margin so a reader knows WHICH field was read or set, on which wire, for that call.
# The two wires size their windows by different mechanisms (D19), and a record that
# states a margin without naming its source cannot be checked (requirement 8).
WINDOW_SOURCE_NATIVE = (
    "options.num_ctx: the window this wire configured for the call (ollama-native)"
)
WINDOW_SOURCE_OPENAI = (
    "ollama ps context_length: the server's loaded window, read on this wire "
    "before each call (ollama-openai)"
)


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
    # WHERE that window came from: one of the WINDOW_SOURCE_* constants above, naming
    # the field this wire read or set for THIS call. A window with no source is a
    # number a reader cannot trace, so the ledger refuses one.
    context_source: str | None = None


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
    # The distinct window sources the recorded calls named (the WINDOW_SOURCE_*
    # constants). The margin above is a number; this is what it was measured against,
    # per wire, and a run may mix wires — so it is a set, filled under the lock with
    # the counters and written into every record through CallStats.
    window_sources: set[str] = field(default_factory=set)
    # Reviewers and the verifier record from worker threads; three unguarded `+=`
    # would lose counts under contention, and a lost count is a wrong published number.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def record(self, response: ChatResponse) -> None:
        context, source = response.context, response.context_source
        if context is not None and source is None:
            raise ValueError(
                f"{response.model}: a call reported a {context:,}-token window with no "
                "context_source. A window nobody can trace back to the field it was read "
                "from is not a run condition, which is the gap round 3 found; every wire "
                "that knows its window says where it read it."
            )
        with self._lock:
            self.prompt_tokens += response.prompt_tokens
            self.completion_tokens += response.completion_tokens
            self.calls += 1
            self.largest_prompt_tokens = max(self.largest_prompt_tokens, response.prompt_tokens)
            if context is not None and source is not None:
                self.smallest_context = (
                    context
                    if self.smallest_context is None
                    else min(self.smallest_context, context)
                )
                headroom = context - response.prompt_tokens
                self.smallest_headroom_tokens = (
                    headroom
                    if self.smallest_headroom_tokens is None
                    else min(self.smallest_headroom_tokens, headroom)
                )
                self.window_sources.add(source)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def window_source_list(self) -> list[str]:
        """The distinct sources this run's windows came from, sorted.

        Sorted and distinct so the same run writes the same list every time; this is
        the value CallStats puts in the record.
        """
        return sorted(self.window_sources)
