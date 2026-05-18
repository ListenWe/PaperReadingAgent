from dataclasses import dataclass, field
from typing import Literal


@dataclass
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class ChatResponse:
    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""


@dataclass
class ProviderConfig:
    provider: Literal["claude", "openai"] = "claude"
    model: str = "claude-sonnet-4-6"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.3
    top_p: float = 1.0
