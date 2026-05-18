from ..llm.types import Message


class ContextManager:
    def __init__(
        self,
        model_token_limit: int = 200000,
        reserved_output_tokens: int = 4096,
    ) -> None:
        self.model_token_limit = model_token_limit
        self.reserved_output = reserved_output_tokens
        self.available = model_token_limit - reserved_output_tokens

    def split_by_token_limit(self, text: str, max_tokens: int | None = None) -> list[str]:
        limit = max_tokens or self.available
        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            if current_tokens + para_tokens > limit and current:
                chunks.append("\n\n".join(current))
                current = [para]
                current_tokens = para_tokens
            else:
                current.append(para)
                current_tokens += para_tokens

        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def fit_messages_in_context(self, messages: list[Message]) -> list[Message]:
        # Always keep the system message and the last few exchanges
        system_msgs = [m for m in messages if m.role == "system"]
        chat_msgs = [m for m in messages if m.role != "system"]

        total = self._count_tokens(messages)
        if total <= self.available:
            return messages

        # Truncate oldest chat messages first, keeping at least last 4 exchanges
        keep_count = 4  # at least 4 chat messages (2 Q&A pairs)
        while len(chat_msgs) > keep_count:
            chat_msgs = chat_msgs[2:]  # remove oldest Q&A pair
            total = self._count_tokens(system_msgs + chat_msgs)
            if total <= self.available:
                break

        return system_msgs + chat_msgs

    def _count_tokens(self, messages: list[Message]) -> int:
        return sum(self._estimate_tokens(m.content) for m in messages)

    def _estimate_tokens(self, text: str) -> int:
        chinese_chars = sum(1 for c in text if "一" <= c <= "鿿")
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)
