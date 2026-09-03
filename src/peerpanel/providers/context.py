"""The context-window contract: a prompt is never silently cut.

Ollama serves each model inside a fixed window (`num_ctx`). A prompt that does
not fit is not refused. The runner logs a warning the caller never sees, keeps
the first four tokens, drops the middle and evaluates the tail — on a 4,096
window the prompt comes back as 2,050 tokens (`llm/llama_server.go`,
`contextShiftPromptLimit`: `numCtx - (numCtx - numKeep) / 2`). The system
prompt is the first thing to go. The OpenAI-compatible endpoint cannot set the
window at all, and Ollama's default is 4,096.

This project ran that way, unknowingly, for every model call above 4,096
tokens until 1 September 2026 (DECISIONS.md D19). Every call now passes two
rules, on both wires:

1. SIZE before the call. A window is chosen from an upper bound on the prompt's
   tokens plus the output budget (the native wire sets it per call; the OpenAI
   wire reads the loaded model's window and refuses a prompt it cannot hold).
2. CHECK after the call. A `prompt_tokens` count that is impossibly low for the
   text sent is the truncation signature, and the call raises instead of
   returning an answer the model never read. What this catches is a served
   window far smaller than the one asked for — the case that caused every
   withdrawn record, where a 4,100-token prompt met a 4,096-token default and
   came back as 2,050. What it cannot catch is a prompt that overruns the window
   it was correctly given by a little: the runner cuts to about half the window,
   and half of a window this prompt nearly filled still reads as a plausible
   ratio. That case is prevented by the sizing bound's margin rather than
   detected, which is why the bound is set from the worst measured prompt class
   and not the typical one.

There is no tokenizer here (Ollama exposes no tokenize endpoint), so both rules
are bounds, measured against real prompts by asking the model to read one and
generate a single token. Reviewer prompts read 3.49 characters per token on qwen2
and 3.90 on llama3.1; extraction prompts, whose JSON candidate lists tokenise far
worse than prose, read 2.71 at worst in CI and 2.21 at worst in the demo corpus.
Sizing assumes 1.5 — well under all of it, because every time this project looked
at denser text the true ratio fell again.

The signature check assumes 6, and that threshold sits between two populations
that overlap at one edge. No measured prompt reads above 4.4, so nothing genuine
trips it. A prompt served in a window far smaller than the one it was sized for
reads about ten characters per token and trips it easily — the case every
withdrawn record came from. A prompt that merely overruns its own window reads
at twice its true ratio, which on the densest class is 5.4, and passes: that case
is prevented by the sizing margin above, not caught here, and the threshold stays
at 6 because a sequence-heavy passage reads high and a lower one would raise on
healthy calls. Windows are powers of two from the floor, so a run uses at
most four distinct sizes; the runner reloads whenever consecutive calls ask for
different ones, which is why the roles with long prompts are kept on one wire
(D19). A prompt bigger than the ceiling is a budgeting defect upstream and is
reported as one, never squeezed.
"""

from __future__ import annotations

# Sizing bound: tokens <= chars / 1.5 for any prompt this project sends.
#
# This constant was wrong twice, in the same direction, and the history is the argument
# for where it now sits. It began at 3.0, from a ratio measured on prose. Measuring the
# extraction prompt — a passage followed by a JSON list of candidate terms, thick with
# gene symbols and accession numbers — gave 2.71, so it moved to 2.5. Then the largest
# prompt in the demo corpus measured 2.21 and overran its window by 498 tokens: a prompt
# the code had just sized would have been silently cut, which is the exact defect D19
# exists to prevent.
#
# Measured with num_predict=1 (chars / prompt_eval_count): reviewer prompts 3.49 (qwen2)
# and 3.90 (llama3.1); extraction prompts 4.40 at their thinnest, 3.85 median, 2.71 worst
# in CI, 2.21 worst in the demo corpus. The ratio is a property of the TEXT, not the
# model, so each look at denser text moved it down again — which is the reason to stop
# tracking the measurements and sit well under all of them.
#
# 1.5 is 32% below the worst ever measured here. It changes no window for a typical
# prompt (a 9,420-character extraction prompt and a 9,965-character reviewer prompt both
# still take 8,192); it moves only the largest prompts up to 16,384, where the biggest
# prompt either corpus contains — 16,959 characters, 7,666 real tokens — has half the
# window spare. The cost is KV memory on the largest calls, not time.
CHARS_PER_TOKEN_SIZING = 1.5
# Truncation signature: a genuine count is never below chars / 6 (docstring math).
CHARS_PER_TOKEN_SIGNATURE = 6.0
# Chat template, role markers and the schema-shaped preamble a wire may add.
TEMPLATE_OVERHEAD_TOKENS = 256
# Ollama's own default window, and the smallest this project asks for.
CONTEXT_FLOOR = 4096
# qwen2:7b's trained window; llama3.1 goes further but this machine's memory
# does not (DECISIONS.md D7 measured the spill).
CONTEXT_CEILING = 32768


class PromptTooLarge(RuntimeError):
    """The prompt plus its output budget cannot fit the largest window allowed."""


class ContextTooSmall(RuntimeError):
    """The wire's window is fixed and smaller than this call needs."""


class PromptTruncated(RuntimeError):
    """The server's prompt count is the truncation signature: the model read a tail."""


def prompt_chars(system: str, user: str) -> int:
    return len(system) + len(user)


def tokens_upper_bound(chars: int) -> int:
    return int(chars / CHARS_PER_TOKEN_SIZING) + TEMPLATE_OVERHEAD_TOKENS


def context_needed(system: str, user: str, max_tokens: int) -> int:
    """The smallest window (a power of two, at least the floor) that holds the
    prompt's upper bound plus the whole output budget, with the one token Ollama
    reserves (`fullPromptLimit = num_ctx - 1`).

    Refuses above the ceiling rather than returning a window that cannot be served:
    this used to hand back 65,536 for a large enough prompt, which no caller could
    honour and which the power-of-two test happily accepted. The ceiling is a rule,
    so it lives with the computation and not only in the wrapper that callers happen
    to use — `context_for_call` re-raises with the model named.
    """
    needed = tokens_upper_bound(prompt_chars(system, user)) + max_tokens + 1
    if needed > CONTEXT_CEILING:
        raise PromptTooLarge(
            f"a {prompt_chars(system, user):,}-character prompt with a {max_tokens:,}-token "
            f"output budget needs a {needed:,}-token window; the ceiling is "
            f"{CONTEXT_CEILING:,}. Budget the prompt at its source."
        )
    window = CONTEXT_FLOOR
    while window < needed:
        window *= 2
    return window


def context_for_call(system: str, user: str, max_tokens: int, *, model: str) -> int:
    """The window for one call — `context_needed`, with the model named in the refusal."""
    needed = tokens_upper_bound(prompt_chars(system, user)) + max_tokens + 1
    if needed > CONTEXT_CEILING:
        raise PromptTooLarge(
            f"{model}: a {prompt_chars(system, user):,}-character prompt with a "
            f"{max_tokens:,}-token output budget needs a {needed:,}-token window; the "
            f"ceiling is {CONTEXT_CEILING:,}. Budget the prompt at its source."
        )
    return context_needed(system, user, max_tokens)


def check_not_truncated(prompt_tokens: int, chars: int, *, model: str, context: int | None) -> None:
    """Raise when the server counted far fewer prompt tokens than the text could hold.

    A count below chars / 6 cannot be a real tokenisation of the text; it is what
    Ollama reports after cutting the prompt to fit a too-small window.
    """
    if prompt_tokens * CHARS_PER_TOKEN_SIGNATURE < chars:
        window = f"a {context:,}-token window" if context else "its window"
        raise PromptTruncated(
            f"{model} reported {prompt_tokens:,} prompt tokens for a {chars:,}-character "
            f"prompt: the server cut the prompt to fit {window} and the model read only "
            "the tail. The call is refused rather than returned. "
            "Native wire: the window is sized per call, so this is a sizing-bound "
            "failure — report it. OpenAI wire: start Ollama with a larger window "
            "(OLLAMA_CONTEXT_LENGTH=16384 or more) or route the call through the "
            "native wire."
        )


def assert_fits(
    context: int, system: str, user: str, max_tokens: int, *, model: str, remedy: str
) -> None:
    """Raise ContextTooSmall when a known, fixed window cannot hold this call."""
    needed = context_needed(system, user, max_tokens)
    if needed > context:
        raise ContextTooSmall(
            f"{model} is loaded with a {context:,}-token window; this call needs "
            f"{needed:,} ({prompt_chars(system, user):,} characters of prompt plus a "
            f"{max_tokens:,}-token output budget). {remedy}"
        )
