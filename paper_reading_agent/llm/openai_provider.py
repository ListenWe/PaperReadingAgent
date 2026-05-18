from collections.abc import Iterator

from tenacity import retry, stop_after_attempt, wait_exponential

from .base import BaseLLMProvider
from .types import ChatResponse, Message, ProviderConfig

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        if not HAS_OPENAI:
            raise ImportError("openai package is required. Install with: pip install openai")
        client_kwargs = {"api_key": config.api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        self._client = OpenAI(**client_kwargs)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def chat(self, messages: list[Message]) -> ChatResponse:
        api_messages = self._convert_messages(messages)
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=api_messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            input_tokens=getattr(response.usage, "prompt_tokens", 0),
            output_tokens=getattr(response.usage, "completion_tokens", 0),
            finish_reason=getattr(choice.finish_reason, "", ""),
        )

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
    def stream_chat(self, messages: list[Message]) -> Iterator[str]:
        api_messages = self._convert_messages(messages)
        stream = self._client.chat.completions.create(
            model=self.config.model,
            messages=api_messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in messages]
