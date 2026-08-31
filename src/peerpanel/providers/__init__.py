"""Provider adapters behind the ChatProvider / EmbedProvider seams."""

from .anthropic_wire import AnthropicChat, MissingAPIKeyError, build_request
from .base import ChatProvider, ChatResponse, EmbedProvider, TokenLedger
from .embed_model2vec import Model2VecEmbed
from .ollama_native import OllamaNativeChat, OllamaNativeEmbed
from .ollama_openai import OllamaOpenAIChat, OllamaOpenAIEmbed

__all__ = [
    "AnthropicChat",
    "ChatProvider",
    "ChatResponse",
    "EmbedProvider",
    "MissingAPIKeyError",
    "Model2VecEmbed",
    "OllamaNativeChat",
    "OllamaNativeEmbed",
    "OllamaOpenAIChat",
    "OllamaOpenAIEmbed",
    "TokenLedger",
    "build_request",
]
