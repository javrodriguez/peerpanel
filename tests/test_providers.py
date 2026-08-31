"""Provider seam tests.

Wire-shape tests run everywhere. Live tests carry the `ollama` marker (need a
local Ollama with the pinned models) or `network` (HuggingFace download) and
are deselected in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from peerpanel.providers import (
    AnthropicChat,
    ChatProvider,
    ChatResponse,
    EmbedProvider,
    MissingAPIKeyError,
    OllamaNativeChat,
    OllamaNativeEmbed,
    OllamaOpenAIChat,
    OllamaOpenAIEmbed,
    TokenLedger,
    build_request,
)
from peerpanel.providers.anthropic_wire import STRUCTURED_TOOL_NAME

SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string"}},
    "required": ["verdict"],
}


class TestAnthropicWireShape:
    """The adapter is UNEXERCISED (no key on this machine, none requested).

    Only the outgoing payload is asserted — never a response body, because no
    real response has ever been recorded.
    """

    def test_plain_request_shape(self) -> None:
        body = build_request(
            model="claude-sonnet-5", system="sys", user="usr", max_tokens=99, temperature=0.5
        )
        assert body == {
            "model": "claude-sonnet-5",
            "max_tokens": 99,
            "temperature": 0.5,
            "system": "sys",
            "messages": [{"role": "user", "content": "usr"}],
        }

    def test_structured_request_forces_the_schema_tool(self) -> None:
        body = build_request(model="m", system="s", user="u", json_schema=SCHEMA)
        assert body["tools"] == [
            {
                "name": STRUCTURED_TOOL_NAME,
                "description": "Return the structured result.",
                "input_schema": SCHEMA,
            }
        ]
        assert body["tool_choice"] == {"type": "tool", "name": STRUCTURED_TOOL_NAME}

    def test_send_path_refuses_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(MissingAPIKeyError):
            AnthropicChat().chat(system="s", user="u")


class TestTokenLedger:
    def test_records_and_totals(self) -> None:
        ledger = TokenLedger()
        ledger.record(ChatResponse(text="a", model="m", prompt_tokens=10, completion_tokens=5))
        ledger.record(ChatResponse(text="b", model="m", prompt_tokens=3, completion_tokens=2))
        assert ledger.calls == 2
        assert ledger.prompt_tokens == 13
        assert ledger.completion_tokens == 7
        assert ledger.total_tokens == 20


class TestProtocolConformance:
    """Every adapter satisfies its seam Protocol (construction needs no server)."""

    def test_chat_adapters(self) -> None:
        assert isinstance(OllamaOpenAIChat("llama3.1:8b"), ChatProvider)
        assert isinstance(OllamaNativeChat("llama3.1:8b"), ChatProvider)
        assert isinstance(AnthropicChat(), ChatProvider)

    def test_embed_adapters(self) -> None:
        assert isinstance(OllamaOpenAIEmbed(), EmbedProvider)
        assert isinstance(OllamaNativeEmbed(), EmbedProvider)


PROMPT = dict(system="You are terse.", user="Name the yeast species S. cerevisiae's genus.")


@pytest.mark.ollama
class TestLiveSeams:
    """The seam exercised for real: the SAME prompt through BOTH wire protocols."""

    def test_same_prompt_both_chat_wires(self) -> None:
        for provider in (OllamaOpenAIChat("llama3.1:8b"), OllamaNativeChat("llama3.1:8b")):
            resp = provider.chat(**PROMPT)  # type: ignore[arg-type]
            assert "Saccharomyces" in resp.text
            assert resp.prompt_tokens > 0
            assert resp.completion_tokens > 0

    def test_second_model_family_on_both_wires(self) -> None:
        for provider in (OllamaOpenAIChat("qwen2:7b"), OllamaNativeChat("qwen2:7b")):
            resp = provider.chat(**PROMPT)  # type: ignore[arg-type]
            assert resp.text.strip()
            assert resp.completion_tokens > 0

    def test_embeddings_normalised_and_deterministic(self) -> None:
        embed = OllamaOpenAIEmbed()
        a = embed.embed(["sulfur metabolism in budding yeast"])
        b = embed.embed(["sulfur metabolism in budding yeast"])
        assert a.shape == (1, 768)
        assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-3)
        assert np.array_equal(a, b)

    def test_both_embed_wires_agree(self) -> None:
        text = ["methionine biosynthesis"]
        via_openai = OllamaOpenAIEmbed().embed(text)
        via_native = OllamaNativeEmbed().embed(text)
        assert np.allclose(via_openai, via_native, atol=1e-5)


@pytest.mark.network
class TestModel2Vec:
    def test_pure_cpu_backend(self) -> None:
        from peerpanel.providers import Model2VecEmbed

        embed = Model2VecEmbed()
        out = embed.embed(["gene expression", "sulfate assimilation"])
        assert out.shape == (2, 256)
        assert out.dtype == np.float32
