from abc import ABC, abstractmethod
from collections.abc import Iterator

from .types import ChatResponse, Message, ProviderConfig


class BaseLLMProvider(ABC):
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @abstractmethod
    def chat(self, messages: list[Message]) -> ChatResponse:
        """Send messages and get a complete response."""

    @abstractmethod
    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        """Send messages and stream response tokens."""

    def estimate_tokens(self, text: str) -> int:
        """Rough token count: ~4 chars per token for English, ~1.5 for Chinese."""
        chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    def count_messages_tokens(self, messages: list[Message]) -> int:
        return sum(self.estimate_tokens(m.content) for m in messages)
