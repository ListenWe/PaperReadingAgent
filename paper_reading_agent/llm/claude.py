from collections.abc import Iterator

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseLLMProvider
from .types import ChatResponse, Message, ProviderConfig

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


class ClaudeProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        if not HAS_ANTHROPIC:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")
        client_kwargs = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._client = anthropic.Anthropic(**client_kwargs)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def chat(self, messages: list[Message]) -> ChatResponse:
        system_msg, api_messages = self._convert_messages(messages)
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": api_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg

        response = self._client.messages.create(**kwargs)

        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return ChatResponse(
            content=content,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason or "",
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        system_msg, api_messages = self._convert_messages(messages)
        kwargs = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": api_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        system_msg = ""
        api_messages: list[dict] = []
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
            else:
                api_messages.append({"role": msg.role, "content": msg.content})
        return system_msg, api_messages
