from collections.abc import Iterator
from pathlib import Path

from ..config import AppConfig
from ..llm.factory import create_llm_provider
from ..llm.types import Message, ProviderConfig
from ..pdf_parser.models import Paper
from ..pdf_parser.parser import PDFParser
from .context_manager import ContextManager
from .prompts import QA_SYSTEM_PROMPT, REPORT_SYNTHESIS_PROMPT, SECTION_SUMMARY_PROMPT


class PaperReadingAgent:
    def __init__(self, config: AppConfig | None = None) -> None:
        self._config = config or AppConfig.from_env()
        self._llm = create_llm_provider(ProviderConfig(
            provider=self._config.llm.provider,
            model=self._config.llm.model,
            api_key=self._config.llm.api_key,
            base_url=self._config.llm.base_url,
            max_tokens=self._config.llm.max_tokens,
            temperature=self._config.llm.temperature,
            top_p=self._config.llm.top_p,
        ))
        self._paper: Paper | None = None
        self._context_manager = ContextManager(
            model_token_limit=self._config.max_context_tokens,
        )
        self._conversation_history: list[Message] = []
        self._section_summaries: dict[str, str] = {}
        self._report: str = ""

    @property
    def paper(self) -> Paper | None:
        return self._paper

    @property
    def report(self) -> str:
        return self._report

    @property
    def history(self) -> list[Message]:
        return self._conversation_history

    def load_paper(self, pdf_path: str | Path) -> Paper:
        parser = PDFParser()
        self._paper = parser.parse(pdf_path)
        self._section_summaries = {}
        self._conversation_history = []
        self._report = ""
        return self._paper

    def generate_report(self) -> str:
        if self._paper is None:
            raise ValueError("No paper loaded. Call load_paper() first.")

        # Phase 1: Summarize each section
        self._section_summaries = {}
        major_sections = [s for s in self._paper.sections if s.level <= 2]

        for section in major_sections:
            if not section.content.strip():
                continue
            summary = self._summarize_section(section.title, section.content)
            self._section_summaries[section.title] = summary

        # Phase 2: Synthesize final report
        authors_str = ", ".join(self._paper.authors[:5])
        if len(self._paper.authors) > 5:
            authors_str += f" 等 ({len(self._paper.authors)} authors)"

        summaries_text = "\n\n---\n\n".join(
            f"### {title}\n{summary}" for title, summary in self._section_summaries.items()
        )

        synthesis_prompt = REPORT_SYNTHESIS_PROMPT.format(
            title=self._paper.title or "Unknown Title",
            authors=authors_str or "Unknown",
            abstract=self._paper.abstract or "No abstract available.",
            section_summaries=summaries_text,
        )

        messages = [
            Message(role="system", content="你是一位资深的学术论文评审专家，擅长深度解读和批判性分析学术论文。请用中文输出。"),
            Message(role="user", content=synthesis_prompt),
        ]

        response = self._llm.chat(messages)
        self._report = response.content
        return self._report

    def generate_report_stream(self) -> Iterator[str]:
        if self._paper is None:
            raise ValueError("No paper loaded. Call load_paper() first.")

        # Phase 1: Summarize each section (non-streaming, needed for synthesis)
        self._section_summaries = {}
        major_sections = [s for s in self._paper.sections if s.level <= 2]

        for section in major_sections:
            if not section.content.strip():
                continue
            summary = self._summarize_section(section.title, section.content)
            self._section_summaries[section.title] = summary

        # Phase 2: Stream the synthesis
        authors_str = ", ".join(self._paper.authors[:5])
        if len(self._paper.authors) > 5:
            authors_str += f" 等 ({len(self._paper.authors)} authors)"

        summaries_text = "\n\n---\n\n".join(
            f"### {title}\n{summary}" for title, summary in self._section_summaries.items()
        )

        synthesis_prompt = REPORT_SYNTHESIS_PROMPT.format(
            title=self._paper.title or "Unknown Title",
            authors=authors_str or "Unknown",
            abstract=self._paper.abstract or "No abstract available.",
            section_summaries=summaries_text,
        )

        messages = [
            Message(role="system", content="你是一位资深的学术论文评审专家，擅长深度解读和批判性分析学术论文。请用中文输出。"),
            Message(role="user", content=synthesis_prompt),
        ]

        self._report = ""
        for chunk in self._llm.stream_chat(messages):
            self._report += chunk
            yield chunk

    def ask_question(self, question: str) -> str:
        if self._paper is None:
            raise ValueError("No paper loaded. Call load_paper() first.")

        full_text = self._get_full_paper_text()
        sections_list = self._build_sections_list()

        system_prompt = QA_SYSTEM_PROMPT.format(
            title=self._paper.title or "Unknown Title",
            abstract=self._paper.abstract or "No abstract available.",
            sections=sections_list,
            context=full_text,
        )

        messages: list[Message] = [Message(role="system", content=system_prompt)]
        messages.extend(self._conversation_history[-10:])
        messages.append(Message(role="user", content=question))

        messages = self._context_manager.fit_messages_in_context(messages)
        response = self._llm.chat(messages)

        self._conversation_history.append(Message(role="user", content=question))
        self._conversation_history.append(Message(role="assistant", content=response.content))

        return response.content

    def ask_question_stream(self, question: str) -> Iterator[str]:
        if self._paper is None:
            raise ValueError("No paper loaded. Call load_paper() first.")

        full_text = self._get_full_paper_text()
        sections_list = self._build_sections_list()

        system_prompt = QA_SYSTEM_PROMPT.format(
            title=self._paper.title or "Unknown Title",
            abstract=self._paper.abstract or "No abstract available.",
            sections=sections_list,
            context=full_text,
        )

        messages: list[Message] = [Message(role="system", content=system_prompt)]
        messages.extend(self._conversation_history[-10:])
        messages.append(Message(role="user", content=question))

        messages = self._context_manager.fit_messages_in_context(messages)

        full_response = ""
        for chunk in self._llm.stream_chat(messages):
            full_response += chunk
            yield chunk

        self._conversation_history.append(Message(role="user", content=question))
        self._conversation_history.append(Message(role="assistant", content=full_response))

    def reset_conversation(self) -> None:
        self._conversation_history = []

    def _summarize_section(self, title: str, content: str) -> str:
        prompt = SECTION_SUMMARY_PROMPT.format(section_content=content)
        messages = [
            Message(role="system", content="你是一位学术论文分析专家。请用中文给出简洁准确的总结。"),
            Message(role="user", content=prompt),
        ]
        response = self._llm.chat(messages)
        return response.content

    def _build_sections_list(self) -> str:
        if self._paper is None:
            return "无"
        lines = []
        for s in self._paper.sections:
            prefix = "  " * (s.level - 1) + "- "
            lines.append(f"{prefix}{s.title} (第{s.start_page+1}-{s.end_page+1}页)")
        return "\n".join(lines)

    def _get_full_paper_text(self) -> str:
        """Return the full paper text (all sections), excluding references."""
        if self._paper is None:
            return ""
        parts = []
        if self._paper.abstract:
            parts.append(f"## 摘要\n\n{self._paper.abstract}\n")
        for section in self._paper.sections:
            if section.title.lower().strip() in ("abstract", "摘要"):
                continue
            parts.append(f"## {section.title}\n\n{section.content}\n")
        return "\n".join(parts)
