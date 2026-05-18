from .base import BaseLLMProvider
from .claude import ClaudeProvider
from .openai_provider import OpenAIProvider
from .types import ProviderConfig


def create_llm_provider(config: ProviderConfig) -> BaseLLMProvider:
    if config.provider == "claude":
        return ClaudeProvider(config)
    elif config.provider == "openai":
        return OpenAIProvider(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}. Use 'claude' or 'openai'.")
