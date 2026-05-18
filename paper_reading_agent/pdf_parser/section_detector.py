import re
from dataclasses import dataclass

from .models import Section


@dataclass
class TextBlock:
    text: str
    page_num: int
    x0: float
    y0: float
    x1: float
    y1: float
    font_size: float = 10.0
    is_bold: bool = False


HEADER_PATTERNS: list[tuple[re.Pattern, int]] = [
    # Numbered sections: "1. Introduction", "2.3.1 Model Architecture"
    (re.compile(r"^\d+(?:\.\d+)*\.?\s+[A-Z][^\d]+", re.UNICODE), 1),
    # Roman numeral: "I. Introduction", "IV. Experiments"
    (re.compile(r"^M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})\.\s+[A-Z]", re.UNICODE), 1),
    # Common English headers (case-insensitive, standalone)
    (re.compile(r"^(?:Abstract|Acknowledgments?|References?|Bibliography|Appendix|Appendices)\s*$", re.IGNORECASE), 1),
    # Common Chinese headers
    (re.compile(r"^(?:摘要|引言|介绍|相关工作|背景|方法|实验|结果|讨论|结论|参考文献|致谢|附录)\s*$", re.UNICODE), 1),
]

SECTION_KEYWORDS: dict[str, int] = {
    "abstract": 1,
    "introduction": 1,
    "related work": 1,
    "background": 1,
    "preliminaries": 1,
    "method": 1,
    "approach": 1,
    "model": 2,
    "architecture": 2,
    "experiment": 1,
    "evaluation": 1,
    "result": 1,
    "discussion": 1,
    "analysis": 2,
    "ablation": 2,
    "conclusion": 1,
    "future work": 2,
    "limitation": 2,
    "appendix": 1,
    "references": 1,
    "bibliography": 1,
    "acknowledgment": 1,
    "acknowledgements": 1,
}

REFERENCE_KEYWORDS = {"references", "bibliography", "参考文献"}


class SectionDetector:
    def detect(self, blocks: list[TextBlock]) -> list[Section]:
        if not blocks:
            return []

        avg_font_size = sum(b.font_size for b in blocks) / len(blocks)
        header_candidates = self._find_header_candidates(blocks, avg_font_size)
        sections = self._build_sections(blocks, header_candidates)
        sections = self._assign_levels(sections)
        return sections

    def _find_header_candidates(
        self, blocks: list[TextBlock], avg_font_size: float
    ) -> list[tuple[int, str, int]]:
        candidates: list[tuple[int, str, int]] = []  # (block_index, title, tentative_level)

        for i, block in enumerate(blocks):
            text = block.text.strip()
            if not text or len(text) > 200:
                continue

            level = self._match_header(text)
            if level is not None:
                candidates.append((i, text, level))
                continue

            if block.font_size >= avg_font_size * 1.3 and len(text) < 150:
                keyword_level = self._match_keywords(text)
                if keyword_level is not None:
                    candidates.append((i, text, keyword_level))

        return candidates

    def _match_header(self, text: str) -> int | None:
        for pattern, level in HEADER_PATTERNS:
            if pattern.match(text):
                # Demote numbered subsections
                if level == 1 and re.match(r"^\d+\.\d+", text):
                    return 2
                return level
        return None

    def _match_keywords(self, text: str) -> int | None:
        lower = text.lower().strip("0123456789. ")
        for keyword, level in SECTION_KEYWORDS.items():
            if lower.startswith(keyword) or lower == keyword:
                return level
        return None

    def _build_sections(
        self, blocks: list[TextBlock], candidates: list[tuple[int, str, int]]
    ) -> list[Section]:
        if not candidates:
            full_text = "\n".join(b.text for b in blocks)
            return [Section(title="Full Text", content=full_text, level=1)]

        sections: list[Section] = []

        for idx, (block_idx, title, level) in enumerate(candidates):
            start_block = block_idx
            end_block = candidates[idx + 1][0] if idx + 1 < len(candidates) else len(blocks)
            content = "\n".join(b.text for b in blocks[start_block + 1 : end_block])

            sections.append(Section(
                title=title,
                content=content,
                level=level,
                start_page=blocks[start_block].page_num,
                end_page=blocks[end_block - 1].page_num if end_block > start_block else blocks[start_block].page_num,
            ))

        return sections

    def _assign_levels(self, sections: list[Section]) -> list[Section]:
        for section in sections:
            # Numbered subsections: count leading digits
            match = re.match(r"^(\d+(?:\.\d+)*)", section.title)
            if match:
                depth = match.group(1).count(".") + 1
                section.level = min(depth, 3)
                continue

            lower = section.title.lower().strip()
            for keyword, level in SECTION_KEYWORDS.items():
                if lower.startswith(keyword):
                    section.level = max(section.level, level)
                    break
        return sections

    def find_reference_boundary(self, sections: list[Section]) -> int | None:
        for i, section in enumerate(sections):
            title_lower = section.title.lower().strip("0123456789. ")
            for kw in REFERENCE_KEYWORDS:
                if title_lower.startswith(kw) or title_lower == kw:
                    return i
        return None
