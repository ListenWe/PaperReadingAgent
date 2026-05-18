import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Search for .env in project root and CWD
_env_paths = [
    Path(__file__).resolve().parent.parent / ".env",  # relative to this package
    Path.cwd() / ".env",
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        break
else:
    load_dotenv()  # fallback: let dotenv search CWD


@dataclass
class ProviderConfig:
    provider: Literal["claude", "openai"] = "claude"
    model: str = "claude-sonnet-4-6"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 16384
    temperature: float = 0.3
    top_p: float = 1.0


@dataclass
class AppConfig:
    llm: ProviderConfig = field(default_factory=ProviderConfig)
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_context_tokens: int = 1_000_000
    report_temperature: float = 0.3
    qa_temperature: float = 0.3

    @classmethod
    def from_env(cls) -> "AppConfig":
        llm_provider = os.getenv("LLM_PROVIDER", "claude").lower()
        if llm_provider not in ("claude", "openai"):
            llm_provider = "claude"

        api_key = (
            os.getenv("ANTHROPIC_API_KEY", "")
            if llm_provider == "claude"
            else os.getenv("OPENAI_API_KEY", "")
        )

        base_url = (
            os.getenv("ANTHROPIC_BASE_URL") or None
            if llm_provider == "claude"
            else os.getenv("OPENAI_BASE_URL") or None
        )

        llm_config = ProviderConfig(
            provider=llm_provider,  # type: ignore[arg-type]
            model=os.getenv("LLM_MODEL", "claude-sonnet-4-6"),
            api_key=api_key,
            base_url=base_url,
            temperature=float(os.getenv("REPORT_TEMPERATURE", "0.3")),
        )

        emb_provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
        if emb_provider not in ("local", "openai"):
            emb_provider = "local"

        return cls(
            llm=llm_config,
            embedding_provider=emb_provider,  # type: ignore[arg-type]
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            max_context_tokens=int(os.getenv("MAX_CONTEXT_TOKENS", "1000000")),
            report_temperature=float(os.getenv("REPORT_TEMPERATURE", "0.3")),
            qa_temperature=float(os.getenv("QA_TEMPERATURE", "0.3")),
        )
