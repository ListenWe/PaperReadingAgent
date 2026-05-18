from __future__ import annotations

import re
from pathlib import Path

import fitz

from .models import Paper, Section
from .section_detector import SectionDetector, TextBlock


class PDFParser:
    def __init__(self) -> None:
        self._detector = SectionDetector()

    def parse(self, filepath: str | Path) -> Paper:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"PDF file not found: {filepath}")

        doc = fitz.open(str(filepath))
        try:
            if doc.page_count == 0:
                raise ValueError("PDF has no pages")

            blocks = self._extract_blocks(doc)
            if not blocks:
                raise ValueError(
                    "Could not extract text from PDF. The file may be a scanned document "
                    "(image-based). Please run OCR first with a tool like ocrmypdf."
                )

            blocks = self._sort_blocks(blocks)
            full_text = "\n".join(b.text for b in blocks)
            title = self._extract_title(blocks, doc)
            sections = self._detector.detect(blocks)
            ref_idx = self._detector.find_reference_boundary(sections)

            abstract = ""
            references = ""
            if ref_idx is not None:
                references = "\n".join(s.title + "\n" + s.content for s in sections[ref_idx:])
                content_sections = sections[:ref_idx]
            else:
                content_sections = sections

            # Extract abstract from the first section or before first numbered section
            abstract = self._extract_abstract(content_sections, full_text)

            return Paper(
                title=title,
                authors=self._extract_authors(blocks),
                abstract=abstract,
                sections=content_sections,
                references=references,
                full_text=full_text,
                metadata={
                    "page_count": doc.page_count,
                    "file_name": filepath.name,
                },
            )
        finally:
            doc.close()

    def _extract_blocks(self, doc: fitz.Document) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text_blocks = page.get_text("dict")["blocks"]
            for blk in text_blocks:
                if blk["type"] != 0:
                    continue
                for line in blk["lines"]:
                    text = "".join(span["text"] for span in line["spans"])
                    if not text.strip():
                        continue
                    spans = line["spans"]
                    font_size = max((s["size"] for s in spans), default=10.0)
                    is_bold = any("Bold" in s.get("font", "") for s in spans)
                    bbox = line["bbox"]
                    blocks.append(TextBlock(
                        text=text.strip(),
                        page_num=page_num,
                        x0=bbox[0],
                        y0=bbox[1],
                        x1=bbox[2],
                        y1=bbox[3],
                        font_size=font_size,
                        is_bold=is_bold,
                    ))
        return blocks

    def _sort_blocks(self, blocks: list[TextBlock]) -> list[TextBlock]:
        # Sort by page, then y-coordinate, then x-coordinate
        blocks.sort(key=lambda b: (b.page_num, round(b.y0 / 20) * 20, b.x0))
        return blocks

    def _extract_title(self, blocks: list[TextBlock], doc: fitz.Document) -> str:
        # Title is usually the largest-font text on the first page
        first_page_blocks = [b for b in blocks if b.page_num == 0]
        if not first_page_blocks:
            return ""
        first_page_blocks.sort(key=lambda b: b.font_size, reverse=True)
        best = first_page_blocks[0]
        if best.font_size > 14 and len(best.text) > 10:
            return best.text
        return ""

    def _extract_authors(self, blocks: list[TextBlock]) -> list[str]:
        # Authors are typically on page 0, below title, with medium font size
        first_page = [b for b in blocks if b.page_num == 0]
        if len(first_page) < 2:
            return []
        # Try to find author line below the title
        first_page_sorted = sorted(first_page, key=lambda b: b.y0)
        if len(first_page_sorted) >= 2:
            second = first_page_sorted[1]
            text = second.text.strip()
            # Common author patterns: names separated by commas, superscript numbers
            if len(text) < 300 and re.search(r"[A-Z][a-z]+", text):
                # Split by commas or superscript numbers
                parts = re.split(r",\s*(?:\d+(?:,\d+)*\s*)?", text)
                return [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
        return []

    def _extract_abstract(self, sections: list[Section], full_text: str) -> str:
        for section in sections:
            if section.title.lower().strip() in ("abstract", "摘要"):
                return section.content.strip()
        # Fallback: search for abstract pattern in full text
        match = re.search(
            r"(?:^Abstract\s*\n|^摘要\s*\n)(.*?)(?=\n\d+\.\s|\n(?:Introduction|引言))",
            full_text, re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return ""
